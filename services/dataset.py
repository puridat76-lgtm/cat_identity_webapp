from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from werkzeug.datastructures import FileStorage

from services.gallery import VALID_EXTENSIONS

INVALID_NAME_CHARS = set('/\\:*?"<>|')


@dataclass
class CatRecord:
    id: int
    name: str
    owner: str = ''
    location: str = ''


class DatasetStore:
    def __init__(self, gallery_dir: Path, meta_path: Path):
        self.gallery_dir = gallery_dir
        self.meta_path = meta_path
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

    def list_cats(self) -> list[dict]:
        return [self._serialize(cat) for cat in self._load_records(sync=True)]

    def get_cat(self, cat_id: int) -> dict:
        cat = self._find_record(cat_id, sync=True)
        return self._serialize(cat)

    def summary(self) -> dict:
        cats = self._load_records(sync=True)
        return {
            'cat_count': len(cats),
            'dataset_images': sum(len(self._image_names(cat)) for cat in cats),
        }

    def create_cat(
        self,
        *,
        name: str,
        owner: str = '',
        location: str = '',
        files: Iterable[FileStorage] = (),
    ) -> dict:
        cats = self._load_records(sync=True)
        uploads = self._prepare_uploads(files)
        clean_name = self._normalize_name(name)
        self._ensure_unique_name(cats, clean_name)

        cat = CatRecord(
            id=self._next_id(cats),
            name=clean_name,
            owner=owner.strip(),
            location=location.strip(),
        )
        target_dir = self._cat_dir(cat.name)
        target_dir.mkdir(parents=True, exist_ok=True)
        self._save_uploads(target_dir, uploads)
        cats.append(cat)
        self._write_records(cats)
        return self._serialize(cat)

    def update_cat(
        self,
        cat_id: int,
        *,
        name: str,
        owner: str = '',
        location: str = '',
        files: Iterable[FileStorage] = (),
    ) -> dict:
        cats = self._load_records(sync=True)
        uploads = self._prepare_uploads(files)
        clean_name = self._normalize_name(name)
        cat = self._find_record(cat_id, records=cats)
        self._ensure_unique_name(cats, clean_name, exclude_id=cat_id)

        current_dir = self._cat_dir(cat.name)
        current_dir.mkdir(parents=True, exist_ok=True)

        if clean_name != cat.name:
            new_dir = self._cat_dir(clean_name)
            if clean_name.casefold() == cat.name.casefold():
                temp_dir = self._cat_dir(f'.tmp_{uuid.uuid4().hex}')
                current_dir.rename(temp_dir)
                temp_dir.rename(new_dir)
            else:
                if new_dir.exists():
                    raise ValueError('ชื่อนี้มีอยู่แล้ว')
                current_dir.rename(new_dir)
        else:
            new_dir = current_dir

        cat.name = clean_name
        cat.owner = owner.strip()
        cat.location = location.strip()
        self._save_uploads(new_dir, uploads)
        self._write_records(cats)
        return self._serialize(cat)

    def delete_cat(self, cat_id: int) -> None:
        cats = self._load_records(sync=True)
        cat = self._find_record(cat_id, records=cats)
        target_dir = self._cat_dir(cat.name)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        remaining = [row for row in cats if row.id != cat_id]
        self._write_records(remaining)

    def delete_image(self, cat_id: int, image_name: str) -> dict:
        image_path = self.get_image_path(cat_id, image_name)
        image_path.unlink()
        return self.get_cat(cat_id)

    def get_image_path(self, cat_id: int, image_name: str) -> Path:
        cat = self._find_record(cat_id, sync=True)
        cat_dir = self._cat_dir(cat.name).resolve()
        image_path = (cat_dir / image_name).resolve()
        if image_path.parent != cat_dir or not image_path.is_file():
            raise FileNotFoundError(image_name)
        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            raise FileNotFoundError(image_name)
        return image_path

    def _load_records(self, *, sync: bool) -> list[CatRecord]:
        records = self._read_records()
        dirty = not self.meta_path.exists()
        deduped: list[CatRecord] = []
        seen_names: set[str] = set()

        for record in records:
            normalized = record.name.casefold()
            if normalized in seen_names:
                dirty = True
                continue
            seen_names.add(normalized)
            deduped.append(record)
            cat_dir = self._cat_dir(record.name)
            if not cat_dir.exists():
                cat_dir.mkdir(parents=True, exist_ok=True)
                dirty = True

        if sync:
            next_id = self._next_id(deduped)
            for folder in sorted(self.gallery_dir.iterdir()):
                if not folder.is_dir():
                    continue
                if folder.name.casefold() in seen_names:
                    continue
                deduped.append(CatRecord(id=next_id, name=folder.name))
                seen_names.add(folder.name.casefold())
                next_id += 1
                dirty = True

        deduped.sort(key=lambda row: row.id)
        if dirty:
            self._write_records(deduped)
        return deduped

    def _read_records(self) -> list[CatRecord]:
        if not self.meta_path.exists():
            return []
        with open(self.meta_path, 'r', encoding='utf-8') as file:
            payload = json.load(file)
        return [
            CatRecord(
                id=int(row['id']),
                name=str(row['name']).strip(),
                owner=str(row.get('owner', '')).strip(),
                location=str(row.get('location', '')).strip(),
            )
            for row in payload
            if str(row.get('name', '')).strip()
        ]

    def _write_records(self, records: list[CatRecord]) -> None:
        payload = [asdict(record) for record in sorted(records, key=lambda row: row.id)]
        with open(self.meta_path, 'w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def _serialize(self, cat: CatRecord) -> dict:
        image_names = self._image_names(cat)
        return {
            **asdict(cat),
            'image_count': len(image_names),
            'image_names': image_names,
        }

    def _image_names(self, cat: CatRecord) -> list[str]:
        return [path.name for path in self._iter_images(self._cat_dir(cat.name))]

    def _prepare_uploads(self, files: Iterable[FileStorage]) -> list[tuple[FileStorage, str]]:
        uploads: list[tuple[FileStorage, str]] = []
        for storage in files:
            filename = (storage.filename or '').strip()
            if not filename:
                continue
            extension = Path(filename).suffix.lower()
            if extension not in VALID_EXTENSIONS:
                raise ValueError('รองรับเฉพาะ JPG, PNG, WEBP')
            uploads.append((storage, extension))
        return uploads

    def _save_uploads(self, target_dir: Path, uploads: list[tuple[FileStorage, str]]) -> None:
        for storage, extension in uploads:
            filename = f'{uuid.uuid4().hex}{extension}'
            storage.save(target_dir / filename)

    def _find_record(
        self,
        cat_id: int,
        *,
        records: list[CatRecord] | None = None,
        sync: bool = False,
    ) -> CatRecord:
        if records is None:
            records = self._load_records(sync=sync)
        for record in records:
            if record.id == cat_id:
                return record
        raise KeyError(cat_id)

    def _ensure_unique_name(
        self,
        records: list[CatRecord],
        name: str,
        *,
        exclude_id: int | None = None,
    ) -> None:
        lowered = name.casefold()
        for record in records:
            if exclude_id is not None and record.id == exclude_id:
                continue
            if record.name.casefold() == lowered:
                raise ValueError('ชื่อนี้มีอยู่แล้ว')

    @staticmethod
    def _normalize_name(name: str) -> str:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError('กรอกชื่อแมวก่อน')
        if clean_name in {'.', '..'}:
            raise ValueError('ชื่อไม่ถูกต้อง')
        if any(char in INVALID_NAME_CHARS for char in clean_name):
            raise ValueError('ชื่อมีอักขระที่ใช้ไม่ได้')
        return clean_name

    def _cat_dir(self, name: str) -> Path:
        return self.gallery_dir / name

    @staticmethod
    def _next_id(records: list[CatRecord]) -> int:
        return max((record.id for record in records), default=0) + 1

    @staticmethod
    def _iter_images(folder: Path) -> Iterable[Path]:
        if not folder.exists():
            return []
        return sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
