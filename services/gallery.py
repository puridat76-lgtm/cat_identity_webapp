from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from services.features import FeatureExtractor

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


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
        self.items: list[GalleryItem] = []
        self.last_build_signature: str | None = None
        self._load_saved_index()

    def _load_saved_index(self) -> None:
        npz_path = self.cfg.index_dir / 'gallery_vectors.npz'
        meta_path = self.cfg.index_dir / 'gallery_meta.json'
        manifest_path = self.cfg.index_dir / 'gallery_manifest.json'
        if not npz_path.exists() or not meta_path.exists():
            self.items = []
            self.last_build_signature = None
            return
        data = np.load(npz_path, allow_pickle=True)
        vectors = data['vectors']
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        self.items = [
            GalleryItem(label=row['label'], path=row['path'], split=row['split'], vector=vectors[i])
            for i, row in enumerate(meta)
        ]
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as file:
                payload = json.load(file)
            self.last_build_signature = payload.get('signature')
        else:
            self.last_build_signature = None

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
        split_totals = {
            'gallery': sum(row['split'] == 'gallery' for row in sources),
            'not_cat': sum(row['split'] == 'not_cat' for row in sources),
            'unknown_cat': sum(row['split'] == 'unknown_cat' for row in sources),
        }
        manifest = [row['manifest'] for row in sources]
        total_images = len(sources)
        processed_images = 0
        valid_images = 0
        split_processed = {'gallery': 0, 'not_cat': 0, 'unknown_cat': 0}

        if progress_callback:
            progress_callback(
                {
                    'stage': 'prepare',
                    'total_images': total_images,
                    'processed_images': processed_images,
                    'valid_images': valid_images,
                    'split_totals': split_totals,
                    'split_processed': split_processed.copy(),
                }
            )

        items: list[GalleryItem] = []
        for row in sources:
            item = self._build_item(row['path'], row['label'], row['split'])
            processed_images += 1
            split_processed[row['split']] += 1
            if item:
                items.append(item)
                valid_images += 1
            if progress_callback:
                progress_callback(
                    {
                        'stage': 'extract',
                        'total_images': total_images,
                        'processed_images': processed_images,
                        'valid_images': valid_images,
                        'current_label': row['label'],
                        'current_image': row['path'].name,
                        'split_totals': split_totals,
                        'split_processed': split_processed.copy(),
                    }
                )

        self.items = items
        if progress_callback:
            progress_callback(
                {
                    'stage': 'save',
                    'total_images': total_images,
                    'processed_images': processed_images,
                    'valid_images': valid_images,
                    'split_totals': split_totals,
                    'split_processed': split_processed.copy(),
                }
            )
        self._save_index(manifest=manifest)
        return self.summary()

    def _save_index(self, *, manifest: list[dict]) -> None:
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        vectors = np.stack([item.vector for item in self.items], axis=0) if self.items else np.empty((0, 0), dtype=np.float32)
        meta = [{'label': item.label, 'path': item.path, 'split': item.split} for item in self.items]
        np.savez_compressed(self.cfg.index_dir / 'gallery_vectors.npz', vectors=vectors)
        with open(self.cfg.index_dir / 'gallery_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self.last_build_signature = self._manifest_signature(manifest)
        with open(self.cfg.index_dir / 'gallery_manifest.json', 'w', encoding='utf-8') as file:
            json.dump({'signature': self.last_build_signature, 'files': manifest}, file, ensure_ascii=False, indent=2)

    def is_stale(self) -> bool:
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

    @staticmethod
    def _manifest_signature(manifest: list[dict]) -> str:
        payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

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
