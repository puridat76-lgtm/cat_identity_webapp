from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from services.features import FeatureExtractor

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
INDEX_FORMAT_VERSION = 3


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
        self.items: list[GalleryItem] = []
        self.last_build_signature: str | None = None
        self._saved_index_valid = False
        self._saved_index_version: int | None = None
        self._saved_extractor_signature: str | None = None
        self._saved_full_rows: list[dict] = []
        self._saved_indexed_rows: list[dict] = []
        self._load_saved_index()

    def _load_saved_index(self) -> None:
        self.items = []
        self.last_build_signature = None
        self._saved_index_valid = False
        self._saved_index_version = None
        self._saved_extractor_signature = None
        self._saved_full_rows = []
        self._saved_indexed_rows = []

        npz_path = self.cfg.index_dir / 'gallery_vectors.npz'
        meta_path = self.cfg.index_dir / 'gallery_meta.json'
        manifest_path = self.cfg.index_dir / 'gallery_manifest.json'
        if not npz_path.exists() or not meta_path.exists():
            return

        try:
            data = np.load(npz_path, allow_pickle=True)
            vectors = data['vectors']
            with open(meta_path, 'r', encoding='utf-8') as file:
                meta = json.load(file)
            if len(meta) != len(vectors):
                return
            self.items = [
                GalleryItem(label=row['label'], path=row['path'], split=row['split'], vector=vectors[i])
                for i, row in enumerate(meta)
            ]
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            self.items = []
            return

        if not manifest_path.exists():
            return

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
            if len(indexed_rows) != len(self.items):
                return
            if not self._manifest_rows_align(indexed_rows):
                return
            self._saved_full_rows = full_rows
            self._saved_indexed_rows = indexed_rows
            self._saved_index_valid = True
        except (OSError, ValueError, json.JSONDecodeError):
            self._saved_index_valid = False
            self._saved_full_rows = []
            self._saved_indexed_rows = []

    def summary(self) -> dict:
        known_labels = sorted({item.label for item in self.items if item.split == 'gallery'})
        counts = {
            'gallery_images': sum(item.split == 'gallery' for item in self.items),
            'not_cat_images': sum(item.split == 'not_cat' for item in self.items),
            'unknown_cat_images': sum(item.split == 'unknown_cat' for item in self.items),
            'known_labels': known_labels,
            'gallery_ready': any(item.split == 'gallery' for item in self.items),
        }
        return counts

    def rebuild(self, progress_callback: Callable[[dict], None] | None = None) -> dict:
        sources = self._collect_sources()
        plan = self._build_rebuild_plan(sources)
        total_images = plan['total_operations']
        processed_images = 0
        valid_images = plan['initial_valid_images']
        split_processed = self._empty_split_counts()

        if progress_callback:
            progress_callback(
                {
                    'stage': 'prepare',
                    'total_images': total_images,
                    'processed_images': processed_images,
                    'valid_images': valid_images,
                    'split_totals': plan['split_totals'],
                    'split_processed': split_processed.copy(),
                }
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

        signature = self._save_index(items=new_items, full_manifest=full_manifest, indexed_manifest=indexed_manifest)
        self.items = new_items
        self.last_build_signature = signature
        self._saved_index_valid = True
        self._saved_index_version = INDEX_FORMAT_VERSION
        self._saved_extractor_signature = self.extractor_signature
        self._saved_full_rows = full_manifest
        self._saved_indexed_rows = indexed_manifest
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
        rows.sort(key=lambda x: x['score'], reverse=True)
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
            rows=[
                {'item': item, 'row': row}
                for item, row in zip(self.items, self._saved_indexed_rows)
                if row['path'] not in current_paths
            ]
        )
        invalid_pool = self._build_hash_pool(
            rows=[
                {'row': row}
                for row in self._saved_full_rows
                if row['path'] not in current_paths and row['path'] not in old_indexed_by_path
            ]
        )
        reused_pool_paths: set[str] = set()

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
                reused_pool_paths.add(indexed_match['row']['path'])
                entries[index] = {'action': 'reuse', 'row': source, 'item': indexed_match['item'], 'manifest': manifest_row}
                continue

            invalid_match = self._pop_hash_match(invalid_pool, content_hash)
            if invalid_match:
                reused_pool_paths.add(invalid_match['row']['path'])
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
            if row['path'] not in current_paths and row['path'] not in reused_pool_paths
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
        entries = []
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

    def _save_index(self, *, items: list[GalleryItem], full_manifest: list[dict], indexed_manifest: list[dict]) -> str:
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        signature = self._manifest_signature(full_manifest)
        vectors = np.stack([item.vector for item in items], axis=0) if items else np.empty((0, 0), dtype=np.float32)
        meta = [{'label': item.label, 'path': item.path, 'split': item.split} for item in items]
        manifest_payload = {
            'version': INDEX_FORMAT_VERSION,
            'extractor_signature': self.extractor_signature,
            'signature': signature,
            'files': full_manifest,
            'indexed_files': indexed_manifest,
        }

        npz_path = self.cfg.index_dir / 'gallery_vectors.npz'
        meta_path = self.cfg.index_dir / 'gallery_meta.json'
        manifest_path = self.cfg.index_dir / 'gallery_manifest.json'

        with tempfile.NamedTemporaryFile(dir=self.cfg.index_dir, suffix='.npz', delete=False) as file:
            temp_npz = Path(file.name)
        try:
            np.savez_compressed(temp_npz, vectors=vectors)
            temp_meta = meta_path.with_suffix('.json.tmp')
            temp_manifest = manifest_path.with_suffix('.json.tmp')
            with open(temp_meta, 'w', encoding='utf-8') as file:
                json.dump(meta, file, ensure_ascii=False, indent=2)
            with open(temp_manifest, 'w', encoding='utf-8') as file:
                json.dump(manifest_payload, file, ensure_ascii=False, indent=2)
            temp_npz.replace(npz_path)
            temp_meta.replace(meta_path)
            temp_manifest.replace(manifest_path)
        finally:
            if temp_npz.exists():
                temp_npz.unlink(missing_ok=True)

        return signature

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
            return GalleryItem(label=label, path=str(image_path), split=split, vector=output.vector)
        except (UnidentifiedImageError, OSError, ValueError):
            return None

    def _can_incrementally_rebuild(self) -> bool:
        return (
            self._saved_index_valid
            and self._saved_index_version == INDEX_FORMAT_VERSION
            and self._saved_extractor_signature == self.extractor_signature
            and len(self.items) == len(self._saved_indexed_rows)
            and all(row.get('content_hash') for row in self._saved_full_rows)
        )

    def _manifest_rows_align(self, rows: list[dict]) -> bool:
        return all(
            row['label'] == item.label and row['path'] == item.path and row['split'] == item.split
            for row, item in zip(rows, self.items)
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
        return all(
            current_row[key] == saved_row.get(key)
            for key in ('label', 'split', 'path', 'size', 'mtime_ns')
        )

    @staticmethod
    def _manifest_with_content_hash(manifest_row: dict, content_hash: str | None) -> dict:
        row = dict(manifest_row)
        if content_hash:
            row['content_hash'] = content_hash
        return row

    @staticmethod
    def _build_hash_pool(rows: list[dict]) -> dict[str, list[dict]]:
        pool: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            content_hash = row['row'].get('content_hash')
            if content_hash:
                pool[content_hash].append(row)
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
        return GalleryItem(label=row['label'], path=str(row['path']), split=row['split'], vector=item.vector.copy())

    @staticmethod
    def _source_row(image_path: Path, label: str, split: str) -> dict:
        stat = image_path.stat()
        return {
            'path': image_path,
            'label': label,
            'split': split,
            'manifest': {
                'label': label,
                'split': split,
                'path': str(image_path.resolve()),
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
