from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from services.features import FeatureExtractor

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
INDEX_FORMAT_VERSION = 4
SQLITE_INDEX_NAME = 'gallery_index.sqlite3'


@dataclass
class GalleryItem:
    label: str
    path: str
    split: str
    vector: np.ndarray


class GalleryIndex:
    def __init__(self, cfg):
        self.cfg = cfg
        self.extractor = FeatureExtractor()
        self.extractor_signature = self.extractor.signature
        self.sqlite_path = self.cfg.index_dir / SQLITE_INDEX_NAME
        self.items: list[GalleryItem] = []
        self.last_build_signature: str | None = None
        self._saved_index_valid = False
        self._saved_index_version: int | None = None
        self._saved_extractor_signature: str | None = None
        self._saved_full_rows: list[dict] = []
        self._saved_indexed_rows: list[dict] = []
        self._saved_backend: str | None = None
        self._load_saved_index()

    def _load_saved_index(self) -> None:
        self._reset_saved_state()
        if self._load_sqlite_index():
            return
        self._reset_saved_state()
        self._load_legacy_index()

    def _reset_saved_state(self) -> None:
        self.items = []
        self.last_build_signature = None
        self._saved_index_valid = False
        self._saved_index_version = None
        self._saved_extractor_signature = None
        self._saved_full_rows = []
        self._saved_indexed_rows = []
        self._saved_backend = None

    def _load_sqlite_index(self) -> bool:
        if not self.sqlite_path.exists():
            return False

        try:
            with self._connect_db() as conn:
                if not self._sqlite_schema_exists(conn):
                    return False

                meta = {row['key']: row['value'] for row in conn.execute('SELECT key, value FROM meta')}
                rows = conn.execute(
                    '''
                    SELECT path, label, split, size, mtime_ns, content_hash, valid, vector
                    FROM items
                    ORDER BY path
                    '''
                ).fetchall()
        except (sqlite3.Error, ValueError):
            return False

        self.last_build_signature = meta.get('signature')
        self._saved_index_version = int(meta.get('version', '0'))
        self._saved_extractor_signature = meta.get('extractor_signature')
        self._saved_backend = 'sqlite'

        full_rows: list[dict] = []
        indexed_rows: list[dict] = []
        items: list[GalleryItem] = []

        for row in rows:
            manifest_row = {
                'label': row['label'],
                'split': row['split'],
                'path': row['path'],
                'size': int(row['size']),
                'mtime_ns': int(row['mtime_ns']),
            }
            if row['content_hash']:
                manifest_row['content_hash'] = row['content_hash']
            full_rows.append(manifest_row)

            if int(row['valid']):
                vector_blob = row['vector']
                if vector_blob is None:
                    return False
                vector = np.frombuffer(vector_blob, dtype=np.float32).copy()
                indexed_rows.append(manifest_row)
                items.append(
                    GalleryItem(
                        label=row['label'],
                        path=row['path'],
                        split=row['split'],
                        vector=vector,
                    )
                )

        self._saved_full_rows = full_rows
        self._saved_indexed_rows = indexed_rows
        self.items = items
        self._saved_index_valid = bool(self.last_build_signature)
        return self._saved_index_valid

    def _load_legacy_index(self) -> bool:
        npz_path = self.cfg.index_dir / 'gallery_vectors.npz'
        meta_path = self.cfg.index_dir / 'gallery_meta.json'
        manifest_path = self.cfg.index_dir / 'gallery_manifest.json'
        if not npz_path.exists() or not meta_path.exists():
            return False

        try:
            data = np.load(npz_path, allow_pickle=True)
            vectors = data['vectors']
            with open(meta_path, 'r', encoding='utf-8') as file:
                meta = json.load(file)
            if len(meta) != len(vectors):
                return False
            items = [
                GalleryItem(label=row['label'], path=row['path'], split=row['split'], vector=vectors[i])
                for i, row in enumerate(meta)
            ]
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return False

        if not manifest_path.exists():
            self.items = items
            self._saved_backend = 'legacy'
            return False

        try:
            with open(manifest_path, 'r', encoding='utf-8') as file:
                payload = json.load(file)
            self.last_build_signature = payload.get('signature')
            self._saved_index_version = int(payload.get('version', 0))
            self._saved_extractor_signature = payload.get('extractor_signature')
            full_rows = self._normalize_manifest_rows(payload.get('files', []))
            indexed_rows = payload.get('indexed_files')
            if indexed_rows is None:
                indexed_rows = payload.get('files', [])
            indexed_rows = self._normalize_manifest_rows(indexed_rows)
            if len(indexed_rows) != len(items):
                return False
            if not self._manifest_rows_align(indexed_rows, items):
                return False
        except (OSError, ValueError, json.JSONDecodeError):
            return False

        self.items = items
        self._saved_full_rows = full_rows
        self._saved_indexed_rows = indexed_rows
        self._saved_index_valid = bool(self.last_build_signature)
        self._saved_backend = 'legacy'
        return self._saved_index_valid

    def summary(self) -> dict:
        known_labels = sorted({item.label for item in self.items if item.split == 'gallery'})
        return {
            'gallery_images': sum(item.split == 'gallery' for item in self.items),
            'not_cat_images': sum(item.split == 'not_cat' for item in self.items),
            'unknown_cat_images': sum(item.split == 'unknown_cat' for item in self.items),
            'known_labels': known_labels,
            'gallery_ready': any(item.split == 'gallery' for item in self.items),
        }

    def rebuild(self, progress_callback: Callable[[dict], None] | None = None) -> dict:
        sources = self._collect_sources()
        plan = self._build_rebuild_plan(sources)
        total_images = plan['total_operations']
        processed_images = 0
        valid_images = plan['initial_valid_images']
        split_processed = self._empty_split_counts()

        self._emit_progress(
            progress_callback,
            stage='prepare',
            total_images=total_images,
            processed_images=processed_images,
            valid_images=valid_images,
            split_totals=plan['split_totals'],
            split_processed=split_processed,
        )

        new_items: list[GalleryItem] = []
        full_manifest: list[dict] = []
        indexed_manifest: list[dict] = []

        for entry in plan['entries']:
            action = entry['action']
            manifest_row = entry['manifest']
            full_manifest.append(manifest_row)

            if action == 'keep':
                new_items.append(self._copy_item(entry['item'], entry['row']))
                indexed_manifest.append(manifest_row)
                continue

            if action == 'keep_invalid':
                continue

            row = entry['row']
            processed_images += 1
            split_processed[row['split']] += 1

            if action == 'reuse':
                new_items.append(self._copy_item(entry['item'], row))
                indexed_manifest.append(manifest_row)
                valid_images += 1
                self._emit_progress(
                    progress_callback,
                    stage='sync',
                    total_images=total_images,
                    processed_images=processed_images,
                    valid_images=valid_images,
                    split_totals=plan['split_totals'],
                    split_processed=split_processed,
                    label=row['label'],
                    image_name=row['path'].name,
                )
                continue

            if action == 'skip':
                self._emit_progress(
                    progress_callback,
                    stage='sync',
                    total_images=total_images,
                    processed_images=processed_images,
                    valid_images=valid_images,
                    split_totals=plan['split_totals'],
                    split_processed=split_processed,
                    label=row['label'],
                    image_name=row['path'].name,
                )
                continue

            item = self._build_item(row['path'], row['label'], row['split'])
            if item:
                new_items.append(item)
                indexed_manifest.append(manifest_row)
                valid_images += 1

            self._emit_progress(
                progress_callback,
                stage='extract',
                total_images=total_images,
                processed_images=processed_images,
                valid_images=valid_images,
                split_totals=plan['split_totals'],
                split_processed=split_processed,
                label=row['label'],
                image_name=row['path'].name,
            )

        for row in plan['delete_rows']:
            processed_images += 1
            split_processed[row['split']] += 1
            self._emit_progress(
                progress_callback,
                stage='remove',
                total_images=total_images,
                processed_images=processed_images,
                valid_images=valid_images,
                split_totals=plan['split_totals'],
                split_processed=split_processed,
                label=row['label'],
                image_name=Path(row['path']).name,
            )

        self._emit_progress(
            progress_callback,
            stage='save',
            total_images=total_images,
            processed_images=processed_images,
            valid_images=valid_images,
            split_totals=plan['split_totals'],
            split_processed=split_processed,
        )

        signature = self._save_index(
            plan=plan,
            full_manifest=full_manifest,
            indexed_manifest=indexed_manifest,
            items=new_items,
        )
        self.items = new_items
        self.last_build_signature = signature
        self._saved_index_valid = True
        self._saved_index_version = INDEX_FORMAT_VERSION
        self._saved_extractor_signature = self.extractor_signature
        self._saved_full_rows = full_manifest
        self._saved_indexed_rows = indexed_manifest
        self._saved_backend = 'sqlite'
        return self.summary()

    def is_stale(self) -> bool:
        if not self.last_build_signature:
            return True
        if not self._saved_index_valid:
            return True
        if self._saved_index_version != INDEX_FORMAT_VERSION:
            return True
        if self._saved_extractor_signature != self.extractor_signature:
            return True
        return self.last_build_signature != self._manifest_signature(self.current_manifest())

    def current_manifest(self) -> list[dict]:
        return [row['manifest'] for row in self._collect_sources()]

    def search(self, vector: np.ndarray, split: str, top_k: int = 5) -> list[dict]:
        candidates = [item for item in self.items if item.split == split]
        if not candidates:
            return []
        rows = []
        for item in candidates:
            score = FeatureExtractor.cosine_similarity(vector, item.vector)
            rows.append({'label': item.label, 'score': score, 'path': item.path})
        rows.sort(key=lambda row: row['score'], reverse=True)
        return rows[:top_k]

    def _build_rebuild_plan(self, sources: list[dict]) -> dict:
        if not self._can_incrementally_rebuild():
            return self._build_full_rebuild_plan(sources)

        current_paths = {row['manifest']['path'] for row in sources}
        old_full_by_path = {row['path']: row for row in self._saved_full_rows}
        old_indexed_by_path = {
            row['path']: {'item': item, 'row': row}
            for item, row in zip(self.items, self._saved_indexed_rows)
        }

        entries: list[dict] = []
        pending: list[tuple[int, dict, dict | None, dict | None]] = []
        initial_valid_images = 0

        for source in sources:
            current_row = source['manifest']
            old_full = old_full_by_path.get(current_row['path'])
            old_indexed = old_indexed_by_path.get(current_row['path'])
            if old_full and self._stat_manifest_matches(current_row, old_full):
                manifest_row = self._manifest_with_content_hash(current_row, old_full.get('content_hash'))
                if old_indexed:
                    entries.append({'action': 'keep', 'row': source, 'item': old_indexed['item'], 'manifest': manifest_row})
                    initial_valid_images += 1
                else:
                    entries.append({'action': 'keep_invalid', 'row': source, 'manifest': manifest_row})
                continue
            pending.append((len(entries), source, old_full, old_indexed))
            entries.append({})

        indexed_pool = self._build_hash_pool(
            [
                {'item': item, 'row': row}
                for item, row in zip(self.items, self._saved_indexed_rows)
                if row['path'] not in current_paths
            ]
        )
        invalid_pool = self._build_hash_pool(
            [
                {'row': row}
                for row in self._saved_full_rows
                if row['path'] not in current_paths and row['path'] not in old_indexed_by_path
            ]
        )

        for index, source, old_full, old_indexed in pending:
            current_row = source['manifest']
            content_hash = self._file_content_hash(source['path'])
            manifest_row = self._manifest_with_content_hash(current_row, content_hash)

            if old_full and old_full.get('content_hash') == content_hash:
                if old_indexed:
                    entries[index] = {'action': 'reuse', 'row': source, 'item': old_indexed['item'], 'manifest': manifest_row}
                else:
                    entries[index] = {'action': 'skip', 'row': source, 'manifest': manifest_row}
                continue

            indexed_match = self._pop_hash_match(indexed_pool, content_hash)
            if indexed_match:
                entries[index] = {'action': 'reuse', 'row': source, 'item': indexed_match['item'], 'manifest': manifest_row}
                continue

            invalid_match = self._pop_hash_match(invalid_pool, content_hash)
            if invalid_match:
                entries[index] = {'action': 'skip', 'row': source, 'manifest': manifest_row}
                continue

            entries[index] = {'action': 'extract', 'row': source, 'manifest': manifest_row}

        split_totals = self._empty_split_counts()
        for entry in entries:
            if entry['action'] in {'reuse', 'skip', 'extract'}:
                split_totals[entry['row']['split']] += 1

        delete_rows = [
            row
            for row in self._saved_full_rows
            if row['path'] not in current_paths and row['content_hash'] not in {
                entry['manifest'].get('content_hash')
                for entry in entries
                if entry['action'] in {'reuse', 'skip'} and entry['manifest'].get('content_hash')
            }
        ]
        for row in delete_rows:
            split_totals[row['split']] += 1

        return {
            'entries': entries,
            'delete_rows': delete_rows,
            'split_totals': split_totals,
            'total_operations': sum(split_totals.values()),
            'initial_valid_images': initial_valid_images,
        }

    def _build_full_rebuild_plan(self, sources: list[dict]) -> dict:
        entries: list[dict] = []
        split_totals = self._empty_split_counts()
        for source in sources:
            manifest_row = self._manifest_with_content_hash(source['manifest'], self._file_content_hash(source['path']))
            entries.append({'action': 'extract', 'row': source, 'manifest': manifest_row})
            split_totals[source['split']] += 1
        return {
            'entries': entries,
            'delete_rows': [],
            'split_totals': split_totals,
            'total_operations': sum(split_totals.values()),
            'initial_valid_images': 0,
        }

    def _save_index(
        self,
        *,
        plan: dict,
        full_manifest: list[dict],
        indexed_manifest: list[dict],
        items: list[GalleryItem],
    ) -> str:
        signature = self._manifest_signature(full_manifest)
        item_by_path = {item.path: item for item in items}
        current_paths = {row['path'] for row in full_manifest}
        prune_paths = {row['path'] for row in self._saved_full_rows if row['path'] not in current_paths}

        if self._saved_backend == 'sqlite' and self._can_incrementally_rebuild():
            upsert_rows = [
                self._serialize_db_row(
                    manifest=row['manifest'],
                    item=item_by_path.get(row['manifest']['path']),
                )
                for row in plan['entries']
                if row['action'] not in {'keep', 'keep_invalid'}
            ]
            self._save_sqlite_incremental(upsert_rows=upsert_rows, prune_paths=prune_paths, signature=signature)
        else:
            rows = [
                self._serialize_db_row(manifest=manifest_row, item=item_by_path.get(manifest_row['path']))
                for manifest_row in full_manifest
            ]
            self._save_sqlite_full(rows=rows, signature=signature)

        return signature

    def _save_sqlite_incremental(self, *, upsert_rows: list[dict], prune_paths: set[str], signature: str) -> None:
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        with self._connect_db() as conn:
            self._ensure_schema(conn)
            with conn:
                if prune_paths:
                    conn.executemany('DELETE FROM items WHERE path = ?', [(path,) for path in sorted(prune_paths)])
                if upsert_rows:
                    conn.executemany(
                        '''
                        INSERT INTO items (path, label, split, size, mtime_ns, content_hash, valid, vector)
                        VALUES (:path, :label, :split, :size, :mtime_ns, :content_hash, :valid, :vector)
                        ON CONFLICT(path) DO UPDATE SET
                          label = excluded.label,
                          split = excluded.split,
                          size = excluded.size,
                          mtime_ns = excluded.mtime_ns,
                          content_hash = excluded.content_hash,
                          valid = excluded.valid,
                          vector = excluded.vector
                        ''',
                        upsert_rows,
                    )
                self._write_meta(conn, signature=signature)

    def _save_sqlite_full(self, *, rows: list[dict], signature: str) -> None:
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        with self._connect_db() as conn:
            self._ensure_schema(conn)
            with conn:
                conn.execute('DELETE FROM items')
                if rows:
                    conn.executemany(
                        '''
                        INSERT INTO items (path, label, split, size, mtime_ns, content_hash, valid, vector)
                        VALUES (:path, :label, :split, :size, :mtime_ns, :content_hash, :valid, :vector)
                        ''',
                        rows,
                    )
                self._write_meta(conn, signature=signature)

    def _write_meta(self, conn: sqlite3.Connection, *, signature: str) -> None:
        payload = {
            'version': str(INDEX_FORMAT_VERSION),
            'extractor_signature': self.extractor_signature,
            'signature': signature,
        }
        conn.executemany(
            'INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value',
            payload.items(),
        )

    def _collect_sources(self) -> list[dict]:
        rows: list[dict] = []
        self.cfg.gallery_dir.mkdir(parents=True, exist_ok=True)
        for label_dir in sorted(self.cfg.gallery_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for image_path in self._iter_image_files(label_dir):
                rows.append(self._source_row(image_path, label, 'gallery'))

        rows.extend(self._collect_flat_folder(self.cfg.not_cat_dir, 'not_cat', 'not_cat'))
        rows.extend(self._collect_flat_folder(self.cfg.unknown_cat_dir, 'unknown', 'unknown_cat'))
        return rows

    def _collect_flat_folder(self, folder: Path, label: str, split: str) -> list[dict]:
        folder.mkdir(parents=True, exist_ok=True)
        return [self._source_row(image_path, label, split) for image_path in self._iter_image_files(folder)]

    def _build_item(self, image_path: Path, label: str, split: str) -> GalleryItem | None:
        try:
            image = Image.open(image_path).convert('RGB')
            output = self.extractor.extract(image)
            return GalleryItem(label=label, path=str(image_path.resolve()), split=split, vector=output.vector)
        except (UnidentifiedImageError, OSError, ValueError):
            return None

    def _can_incrementally_rebuild(self) -> bool:
        return (
            self._saved_index_valid
            and self._saved_index_version is not None
            and self._saved_index_version >= 3
            and self._saved_extractor_signature == self.extractor_signature
            and len(self.items) == len(self._saved_indexed_rows)
            and all(row.get('content_hash') for row in self._saved_full_rows)
        )

    @staticmethod
    def _manifest_rows_align(rows: list[dict], items: list[GalleryItem]) -> bool:
        return all(
            row['label'] == item.label and row['path'] == item.path and row['split'] == item.split
            for row, item in zip(rows, items)
        )

    @staticmethod
    def _normalize_manifest_rows(payload: list[dict]) -> list[dict]:
        if not isinstance(payload, list):
            raise ValueError('manifest rows must be a list')
        return [GalleryIndex._normalize_manifest_row(row) for row in payload]

    @staticmethod
    def _normalize_manifest_row(row: dict) -> dict:
        if not isinstance(row, dict):
            raise ValueError('manifest row must be an object')
        normalized = {
            'label': str(row['label']),
            'split': str(row['split']),
            'path': str(row['path']),
            'size': int(row['size']),
            'mtime_ns': int(row['mtime_ns']),
        }
        if row.get('content_hash'):
            normalized['content_hash'] = str(row['content_hash'])
        return normalized

    @staticmethod
    def _manifest_signature(manifest: list[dict]) -> str:
        payload = json.dumps(
            [
                {
                    'label': row['label'],
                    'split': row['split'],
                    'path': row['path'],
                    'size': row['size'],
                    'mtime_ns': row['mtime_ns'],
                }
                for row in manifest
            ],
            ensure_ascii=False,
            sort_keys=True,
        ).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _stat_manifest_matches(current_row: dict, saved_row: dict) -> bool:
        return all(current_row[key] == saved_row.get(key) for key in ('label', 'split', 'path', 'size', 'mtime_ns'))

    @staticmethod
    def _manifest_with_content_hash(manifest_row: dict, content_hash: str | None) -> dict:
        row = dict(manifest_row)
        if content_hash:
            row['content_hash'] = content_hash
        return row

    @staticmethod
    def _build_hash_pool(rows: list[dict]) -> dict[str, list[dict]]:
        pool: dict[str, list[dict]] = {}
        for row in rows:
            content_hash = row['row'].get('content_hash')
            if not content_hash:
                continue
            pool.setdefault(content_hash, []).append(row)
        return pool

    @staticmethod
    def _pop_hash_match(pool: dict[str, list[dict]], content_hash: str) -> dict | None:
        matches = pool.get(content_hash)
        if not matches:
            return None
        value = matches.pop()
        if not matches:
            pool.pop(content_hash, None)
        return value

    @staticmethod
    def _copy_item(item: GalleryItem, row: dict) -> GalleryItem:
        return GalleryItem(label=row['label'], path=str(row['path'].resolve()), split=row['split'], vector=item.vector.copy())

    @staticmethod
    def _source_row(image_path: Path, label: str, split: str) -> dict:
        resolved = image_path.resolve()
        stat = resolved.stat()
        return {
            'path': resolved,
            'label': label,
            'split': split,
            'manifest': {
                'label': label,
                'split': split,
                'path': str(resolved),
                'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns,
            },
        }

    @staticmethod
    def _iter_image_files(folder: Path) -> Iterable[Path]:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                yield path

    @staticmethod
    def _empty_split_counts() -> dict[str, int]:
        return {'gallery': 0, 'not_cat': 0, 'unknown_cat': 0}

    @staticmethod
    def _emit_progress(
        progress_callback: Callable[[dict], None] | None,
        *,
        stage: str,
        total_images: int,
        processed_images: int,
        valid_images: int,
        split_totals: dict,
        split_processed: dict,
        label: str | None = None,
        image_name: str | None = None,
    ) -> None:
        if not progress_callback:
            return
        progress_callback(
            {
                'stage': stage,
                'total_images': total_images,
                'processed_images': processed_images,
                'valid_images': valid_images,
                'current_label': label,
                'current_image': image_name,
                'split_totals': split_totals,
                'split_processed': split_processed.copy(),
            }
        )

    @staticmethod
    def _file_content_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, 'rb') as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _serialize_db_row(*, manifest: dict, item: GalleryItem | None) -> dict:
        return {
            'path': manifest['path'],
            'label': manifest['label'],
            'split': manifest['split'],
            'size': manifest['size'],
            'mtime_ns': manifest['mtime_ns'],
            'content_hash': manifest.get('content_hash'),
            'valid': 1 if item is not None else 0,
            'vector': None if item is None else item.vector.astype(np.float32).tobytes(),
        }

    def _connect_db(self) -> sqlite3.Connection:
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _sqlite_schema_exists(conn: sqlite3.Connection) -> bool:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('meta', 'items')"
        ).fetchall()
        return {row['name'] for row in rows} == {'meta', 'items'}

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            )
            '''
        )
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS items (
              path TEXT PRIMARY KEY,
              label TEXT NOT NULL,
              split TEXT NOT NULL,
              size INTEGER NOT NULL,
              mtime_ns INTEGER NOT NULL,
              content_hash TEXT,
              valid INTEGER NOT NULL,
              vector BLOB
            )
            '''
        )
        conn.execute('CREATE INDEX IF NOT EXISTS idx_items_split ON items(split)')
