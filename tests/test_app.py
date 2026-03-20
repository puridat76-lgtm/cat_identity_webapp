from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app import create_app


def make_pattern_image(color: tuple[int, int, int], accent: tuple[int, int, int], mode: str = 'cat') -> Image.Image:
    image = Image.new('RGB', (320, 320), color=color)
    draw = ImageDraw.Draw(image)
    if mode == 'cat':
        draw.ellipse((60, 50, 260, 250), fill=accent)
        draw.polygon([(90, 80), (130, 10), (170, 90)], fill=accent)
        draw.polygon([(150, 90), (210, 10), (235, 90)], fill=accent)
    elif mode == 'not_cat':
        draw.rectangle((60, 60, 260, 260), fill=accent)
    return image


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def image_bytes(image: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return buf


def setup_dataset(root: Path) -> None:
    save_image(root / 'gallery' / 'Haha' / 'haha_1.png', make_pattern_image((120, 120, 120), (170, 170, 170), mode='cat'))
    save_image(root / 'gallery' / 'Loki' / 'loki_1.png', make_pattern_image((226, 140, 48), (255, 189, 112), mode='cat'))
    save_image(root / 'not_cat' / 'bottle_1.png', make_pattern_image((40, 80, 180), (20, 40, 110), mode='not_cat'))
    save_image(root / 'unknown_cat' / 'other_cat_1.png', make_pattern_image((245, 245, 245), (20, 20, 20), mode='cat'))


def test_status_and_rebuild_and_predict(tmp_path):
    data_dir = tmp_path / 'data'
    setup_dataset(data_dir)
    app = create_app(
        {
            'TESTING': True,
            'DATA_DIR': data_dir,
            'INDEX_DIR': data_dir / 'index',
            'GALLERY_DIR': data_dir / 'gallery',
            'NOT_CAT_DIR': data_dir / 'not_cat',
            'UNKNOWN_CAT_DIR': data_dir / 'unknown_cat',
        }
    )
    client = app.test_client()

    res = client.get('/api/status')
    assert res.status_code == 200
    assert res.get_json()['gallery_ready'] is False

    rebuild = client.post('/api/rebuild')
    assert rebuild.status_code == 200
    rebuilt = rebuild.get_json()
    assert rebuilt['gallery_ready'] is True
    assert rebuilt['gallery_images'] == 2

    query = make_pattern_image((122, 122, 122), (176, 176, 176), mode='cat')
    response = client.post('/api/predict', data={'file': (image_bytes(query), 'test.png')}, content_type='multipart/form-data')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['final_label'] == 'Haha'


def test_not_cat_and_low_quality(tmp_path):
    data_dir = tmp_path / 'data'
    setup_dataset(data_dir)
    app = create_app(
        {
            'TESTING': True,
            'DATA_DIR': data_dir,
            'INDEX_DIR': data_dir / 'index',
            'GALLERY_DIR': data_dir / 'gallery',
            'NOT_CAT_DIR': data_dir / 'not_cat',
            'UNKNOWN_CAT_DIR': data_dir / 'unknown_cat',
        }
    )
    client = app.test_client()
    client.post('/api/rebuild')

    not_cat = make_pattern_image((40, 80, 180), (20, 40, 110), mode='not_cat')
    response = client.post('/api/predict', data={'file': (image_bytes(not_cat), 'not_cat.png')}, content_type='multipart/form-data')
    assert response.status_code == 200
    assert response.get_json()['final_label'] == 'not_cat'

    tiny = Image.new('RGB', (40, 40), color=(100, 100, 100))
    response2 = client.post('/api/predict', data={'file': (image_bytes(tiny), 'tiny.png')}, content_type='multipart/form-data')
    assert response2.status_code == 200
    assert response2.get_json()['final_label'] == 'low_quality'
