from __future__ import annotations

import io
import json
import sqlite3
import time
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


def make_app(data_dir: Path):
    return create_app(
        {
            'TESTING': True,
            'DATA_DIR': data_dir,
            'INDEX_DIR': data_dir / 'index',
            'GALLERY_DIR': data_dir / 'gallery',
            'NOT_CAT_DIR': data_dir / 'not_cat',
            'UNKNOWN_CAT_DIR': data_dir / 'unknown_cat',
            'CATS_META_PATH': data_dir / 'cats.json',
        }
    )


def wait_for_train(client, attempts: int = 40) -> dict:
    final_job = None
    for _ in range(attempts):
        status_response = client.get('/api/train/status')
        assert status_response.status_code == 200
        final_job = status_response.get_json()['job']
        if final_job['status'] in {'completed', 'failed'}:
            break
        time.sleep(0.05)
    assert final_job is not None
    return final_job


def start_and_wait_train(client) -> dict:
    start_response = client.post('/api/train')
    assert start_response.status_code == 202
    payload = start_response.get_json()
    assert payload['started'] is True
    assert payload['job']['status'] == 'running'
    return wait_for_train(client)


def test_status_and_rebuild_and_predict(tmp_path):
    data_dir = tmp_path / 'data'
    setup_dataset(data_dir)
    app = make_app(data_dir)
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


def test_pages_render(tmp_path):
    data_dir = tmp_path / 'data'
    setup_dataset(data_dir)
    app = make_app(data_dir)
    client = app.test_client()

    for path in ['/', '/cats', '/not-cat', '/unknown-cat', '/train', '/predict']:
        response = client.get(path)
        assert response.status_code == 200


def test_not_cat_and_low_quality(tmp_path):
    data_dir = tmp_path / 'data'
    setup_dataset(data_dir)
    app = make_app(data_dir)
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


def test_dataset_crud_and_image_delete(tmp_path):
    data_dir = tmp_path / 'data'
    setup_dataset(data_dir)
    app = make_app(data_dir)
    client = app.test_client()

    listing = client.get('/api/cats')
    assert listing.status_code == 200
    cats = listing.get_json()['cats']
    assert [cat['name'] for cat in cats] == ['Haha', 'Loki']
    assert cats[0]['image_count'] == 1

    create_response = client.post(
        '/api/cats',
        data={
            'name': 'Nova',
            'owner': 'Mint',
            'location': 'Room A',
            'images': (image_bytes(make_pattern_image((80, 80, 80), (200, 200, 200), mode='cat')), 'nova.png'),
        },
        content_type='multipart/form-data',
    )
    assert create_response.status_code == 201
    created = create_response.get_json()['cat']
    assert created['name'] == 'Nova'
    assert created['image_count'] == 1

    update_response = client.put(
        f"/api/cats/{created['id']}",
        data={
            'name': 'Nova Prime',
            'owner': 'Mint',
            'location': 'Room B',
            'images': (image_bytes(make_pattern_image((82, 82, 82), (210, 210, 210), mode='cat')), 'nova_2.png'),
        },
        content_type='multipart/form-data',
    )
    assert update_response.status_code == 200
    updated = update_response.get_json()['cat']
    assert updated['name'] == 'Nova Prime'
    assert updated['location'] == 'Room B'
    assert updated['image_count'] == 2
    assert not (data_dir / 'gallery' / 'Nova').exists()
    assert (data_dir / 'gallery' / 'Nova Prime').exists()

    image_name = updated['images'][0]['name']
    delete_image_response = client.delete(f"/api/cats/{updated['id']}/images/{image_name}")
    assert delete_image_response.status_code == 200
    after_delete_image = delete_image_response.get_json()['cat']
    assert after_delete_image['image_count'] == 1

    status_response = client.get('/api/status')
    assert status_response.status_code == 200
    status_payload = status_response.get_json()
    assert status_payload['index_status'] == 'needs_train'

    rebuild_response = client.post('/api/rebuild')
    assert rebuild_response.status_code == 200
    rebuilt_payload = rebuild_response.get_json()
    assert 'Nova Prime' in rebuilt_payload['known_labels']
    assert rebuilt_payload['gallery_images'] == 3
    assert rebuilt_payload['index_status'] == 'ready'

    delete_cat_response = client.delete(f"/api/cats/{updated['id']}")
    assert delete_cat_response.status_code == 200
    assert not (data_dir / 'gallery' / 'Nova Prime').exists()

    stale_after_delete = client.get('/api/status')
    assert stale_after_delete.status_code == 200
    assert stale_after_delete.get_json()['index_status'] == 'needs_train'

    final_listing = client.get('/api/cats')
    assert [cat['name'] for cat in final_listing.get_json()['cats']] == ['Haha', 'Loki']


def test_reference_sets_and_train_status(tmp_path):
    data_dir = tmp_path / 'data'
    setup_dataset(data_dir)
    app = make_app(data_dir)
    client = app.test_client()

    client.post('/api/rebuild')

    reference_sets_response = client.get('/api/reference-sets')
    assert reference_sets_response.status_code == 200
    reference_sets = {row['key']: row for row in reference_sets_response.get_json()['reference_sets']}
    assert reference_sets['not_cat']['image_count'] == 1
    assert reference_sets['unknown_cat']['image_count'] == 1

    save_image(
        data_dir / 'unknown_cat' / 'manual_unknown.png',
        make_pattern_image((180, 180, 180), (40, 40, 40), mode='cat'),
    )
    stale_status = client.get('/api/status')
    assert stale_status.status_code == 200
    stale_payload = stale_status.get_json()
    assert stale_payload['index_status'] == 'needs_train'
    assert stale_payload['unknown_cat_folder_images'] == 2

    upload_response = client.post(
        '/api/reference-sets/not_cat/images',
        data={
            'images': (image_bytes(make_pattern_image((70, 100, 180), (10, 30, 90), mode='not_cat')), 'ref_not_cat.png'),
        },
        content_type='multipart/form-data',
    )
    assert upload_response.status_code == 201
    uploaded_reference = upload_response.get_json()['reference_set']
    assert uploaded_reference['key'] == 'not_cat'
    assert uploaded_reference['image_count'] == 2
    assert upload_response.get_json()['summary']['index_status'] == 'needs_train'

    rebuild_response = client.post('/api/rebuild')
    assert rebuild_response.status_code == 200

    trained_status = client.get('/api/status')
    assert trained_status.status_code == 200
    trained_payload = trained_status.get_json()
    assert trained_payload['index_status'] == 'ready'
    assert trained_payload['not_cat_images'] == 2
    assert trained_payload['unknown_cat_images'] == 2

    image_name = uploaded_reference['images'][0]['name']
    delete_response = client.delete(f'/api/reference-sets/not_cat/images/{image_name}')
    assert delete_response.status_code == 200
    assert delete_response.get_json()['reference_set']['image_count'] == 1
    assert delete_response.get_json()['summary']['index_status'] == 'needs_train'


def test_train_job_status_endpoint(tmp_path):
    data_dir = tmp_path / 'data'
    setup_dataset(data_dir)
    app = make_app(data_dir)
    client = app.test_client()

    final_job = start_and_wait_train(client)
    assert final_job['status'] == 'completed'
    assert final_job['processed_images'] == 4
    assert final_job['summary']['index_status'] == 'ready'


def test_incremental_train_skips_unchanged_and_processes_only_new_file(tmp_path):
    data_dir = tmp_path / 'data'
    setup_dataset(data_dir)
    app = make_app(data_dir)
    client = app.test_client()

    initial_job = start_and_wait_train(client)
    assert initial_job['processed_images'] == 4
    assert (data_dir / 'index' / 'gallery_index.sqlite3').exists()

    reloaded_app = make_app(data_dir)
    reloaded_client = reloaded_app.test_client()
    no_change_job = start_and_wait_train(reloaded_client)
    assert no_change_job['status'] == 'completed'
    assert no_change_job['processed_images'] == 0
    assert no_change_job['valid_images'] == 4

    save_image(
        data_dir / 'gallery' / 'Haha' / 'haha_2.png',
        make_pattern_image((122, 122, 122), (180, 180, 180), mode='cat'),
    )
    incremental_job = start_and_wait_train(reloaded_client)
    assert incremental_job['status'] == 'completed'
    assert incremental_job['processed_images'] == 1
    assert incremental_job['valid_images'] == 5
    assert incremental_job['summary']['gallery_images'] == 3
    assert incremental_job['summary']['index_status'] == 'ready'


def test_incremental_train_reuses_vectors_for_rename_and_delete(tmp_path):
    data_dir = tmp_path / 'data'
    setup_dataset(data_dir)
    app = make_app(data_dir)
    client = app.test_client()

    start_and_wait_train(client)

    cats = client.get('/api/cats').get_json()['cats']
    haha = next(cat for cat in cats if cat['name'] == 'Haha')

    rename_response = client.put(
        f"/api/cats/{haha['id']}",
        data={'name': 'Haha Prime', 'owner': '', 'location': ''},
        content_type='multipart/form-data',
    )
    assert rename_response.status_code == 200

    rename_job = start_and_wait_train(client)
    assert rename_job['status'] == 'completed'
    assert rename_job['processed_images'] == 1
    assert 'Haha Prime' in rename_job['summary']['known_labels']
    assert 'Haha' not in rename_job['summary']['known_labels']

    updated_cat = rename_response.get_json()['cat']
    image_name = updated_cat['images'][0]['name']
    delete_response = client.delete(f"/api/cats/{updated_cat['id']}/images/{image_name}")
    assert delete_response.status_code == 200

    delete_job = start_and_wait_train(client)
    assert delete_job['status'] == 'completed'
    assert delete_job['processed_images'] == 1
    assert delete_job['summary']['gallery_images'] == 1
    assert delete_job['summary']['known_labels'] == ['Loki']


def test_incremental_train_falls_back_to_full_rebuild_when_saved_state_is_legacy(tmp_path):
    data_dir = tmp_path / 'data'
    setup_dataset(data_dir)
    app = make_app(data_dir)
    client = app.test_client()

    start_and_wait_train(client)

    sqlite_path = data_dir / 'index' / 'gallery_index.sqlite3'
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("UPDATE meta SET value = '1' WHERE key = 'version'")
        conn.commit()

    reloaded_app = make_app(data_dir)
    reloaded_client = reloaded_app.test_client()
    fallback_job = start_and_wait_train(reloaded_client)
    assert fallback_job['status'] == 'completed'
    assert fallback_job['processed_images'] == 4
    assert fallback_job['summary']['index_status'] == 'ready'
