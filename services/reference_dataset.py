from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable

from werkzeug.datastructures import FileStorage

from services.gallery import VALID_EXTENSIONS


class ReferenceDatasetStore:
    def __init__(self, folder: Path):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)

    def list_images(self, limit: int | None = None) -> list[Path]:
        images = sorted(
            (
                path
                for path in self.folder.iterdir()
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
            ),
            key=lambda path: (path.stat().st_mtime, path.name.lower()),
            reverse=True,
        )
        if limit is not None:
            return images[:limit]
        return images

    def summary(self) -> dict:
        return {'image_count': len(self.list_images())}

    def add_images(self, files: Iterable[FileStorage]) -> int:
        uploads = self._prepare_uploads(files)
        for storage, extension in uploads:
            filename = f'{uuid.uuid4().hex}{extension}'
            storage.save(self.folder / filename)
        return len(uploads)

    def delete_image(self, image_name: str) -> None:
        self.get_image_path(image_name).unlink()

    def get_image_path(self, image_name: str) -> Path:
        image_path = (self.folder.resolve() / image_name).resolve()
        if image_path.parent != self.folder.resolve() or not image_path.is_file():
            raise FileNotFoundError(image_name)
        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            raise FileNotFoundError(image_name)
        return image_path

    @staticmethod
    def _prepare_uploads(files: Iterable[FileStorage]) -> list[tuple[FileStorage, str]]:
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
