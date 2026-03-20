from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

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
        self._load_saved_index()

    def _load_saved_index(self) -> None:
        npz_path = self.cfg.index_dir / 'gallery_vectors.npz'
        meta_path = self.cfg.index_dir / 'gallery_meta.json'
        if not npz_path.exists() or not meta_path.exists():
            self.items = []
            return
        data = np.load(npz_path, allow_pickle=True)
        vectors = data['vectors']
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        self.items = [
            GalleryItem(label=row['label'], path=row['path'], split=row['split'], vector=vectors[i])
            for i, row in enumerate(meta)
        ]

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

    def rebuild(self) -> dict:
        items: list[GalleryItem] = []
        items.extend(self._scan_known_gallery())
        items.extend(self._scan_flat_folder(self.cfg.not_cat_dir, 'not_cat', 'not_cat'))
        items.extend(self._scan_flat_folder(self.cfg.unknown_cat_dir, 'unknown', 'unknown_cat'))
        self.items = items
        self._save_index()
        return self.summary()

    def _save_index(self) -> None:
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        vectors = np.stack([item.vector for item in self.items], axis=0) if self.items else np.empty((0, 0), dtype=np.float32)
        meta = [{'label': item.label, 'path': item.path, 'split': item.split} for item in self.items]
        np.savez_compressed(self.cfg.index_dir / 'gallery_vectors.npz', vectors=vectors)
        with open(self.cfg.index_dir / 'gallery_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

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

    def _scan_known_gallery(self) -> list[GalleryItem]:
        items: list[GalleryItem] = []
        self.cfg.gallery_dir.mkdir(parents=True, exist_ok=True)
        for label_dir in sorted(self.cfg.gallery_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for image_path in self._iter_image_files(label_dir):
                item = self._build_item(image_path, label, 'gallery')
                if item:
                    items.append(item)
        return items

    def _scan_flat_folder(self, folder: Path, label: str, split: str) -> list[GalleryItem]:
        folder.mkdir(parents=True, exist_ok=True)
        items = []
        for image_path in self._iter_image_files(folder):
            item = self._build_item(image_path, label, split)
            if item:
                items.append(item)
        return items

    def _build_item(self, image_path: Path, label: str, split: str) -> GalleryItem | None:
        try:
            image = Image.open(image_path).convert('RGB')
            output = self.extractor.extract(image)
            return GalleryItem(label=label, path=str(image_path), split=split, vector=output.vector)
        except (UnidentifiedImageError, OSError, ValueError):
            return None

    @staticmethod
    def _iter_image_files(folder: Path) -> Iterable[Path]:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                yield path
