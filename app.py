from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, url_for
from PIL import Image

from services.dataset import DatasetStore
from services.gallery import GalleryIndex
from services.pipeline import CatIdentityPipeline, PipelineConfig
from services.reference_dataset import ReferenceDatasetStore
from services.train import TrainManager

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / 'data'


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def create_app(test_config: dict | None = None) -> Flask:
    data_dir = _env_path('DATA_DIR', DEFAULT_DATA_DIR)
    index_dir = _env_path('INDEX_DIR', data_dir / 'index')
    gallery_dir = _env_path('GALLERY_DIR', data_dir / 'gallery')
    not_cat_dir = _env_path('NOT_CAT_DIR', data_dir / 'not_cat')
    unknown_cat_dir = _env_path('UNKNOWN_CAT_DIR', data_dir / 'unknown_cat')
    cats_meta_path = _env_path('CATS_META_PATH', data_dir / 'cats.json')

    app = Flask(__name__)
    app.config.update(
        SECRET_KEY='dev',
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        DATA_DIR=data_dir,
        INDEX_DIR=index_dir,
        GALLERY_DIR=gallery_dir,
        NOT_CAT_DIR=not_cat_dir,
        UNKNOWN_CAT_DIR=unknown_cat_dir,
        CATS_META_PATH=cats_meta_path,
    )

    if test_config:
        app.config.update(test_config)

    Path(app.config['DATA_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app.config['INDEX_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app.config['GALLERY_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app.config['NOT_CAT_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app.config['UNKNOWN_CAT_DIR']).mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        gallery_dir=Path(app.config['GALLERY_DIR']),
        not_cat_dir=Path(app.config['NOT_CAT_DIR']),
        unknown_cat_dir=Path(app.config['UNKNOWN_CAT_DIR']),
        index_dir=Path(app.config['INDEX_DIR']),
    )
    dataset_store = DatasetStore(
        gallery_dir=Path(app.config['GALLERY_DIR']),
        meta_path=Path(app.config['CATS_META_PATH']),
    )
    reference_sets = {
        'not_cat': {
            'label': 'ไม่ใช่แมว',
            'folder_label': 'not_cat',
            'store': ReferenceDatasetStore(Path(app.config['NOT_CAT_DIR'])),
        },
        'unknown_cat': {
            'label': 'แมวจากอินเตอร์เน็ต',
            'folder_label': 'unknown_cat',
            'store': ReferenceDatasetStore(Path(app.config['UNKNOWN_CAT_DIR'])),
        },
    }
    gallery_index = GalleryIndex(cfg)
    pipeline = CatIdentityPipeline(cfg, gallery_index)
    app.extensions['dataset_store'] = dataset_store
    app.extensions['reference_sets'] = reference_sets
    app.extensions['gallery_index'] = gallery_index
    app.extensions['pipeline'] = pipeline

    def render_page(template_name: str, *, active_page: str, title: str, **context):
        return render_template(
            template_name,
            active_page=active_page,
            page_title=title,
            **context,
        )

    @app.get('/')
    def index():
        return render_page('cats.html', active_page='cats', title='Cats')

    @app.get('/cats')
    def cats():
        return render_page('cats.html', active_page='cats', title='Cats')

    @app.get('/not-cat')
    def not_cat_page():
        return render_page(
            'reference.html',
            active_page='not_cat',
            title='Not cat',
            reference_key='not_cat',
            page_heading='Not cat',
            page_copy='คลาสอ้างอิงสำหรับรูปที่ไม่ใช่แมว',
        )

    @app.get('/unknown-cat')
    def unknown_cat_page():
        return render_page(
            'reference.html',
            active_page='unknown_cat',
            title='Unknown_cat',
            reference_key='unknown_cat',
            page_heading='Unknown_cat',
            page_copy='คลาสอ้างอิงสำหรับแมวจากอินเตอร์เน็ตหรือแมวที่ยังไม่รู้จัก',
        )

    @app.get('/train')
    def train_page():
        return render_page('train.html', active_page='train', title='Train')

    @app.get('/predict')
    def predict_page():
        return render_page('predict.html', active_page='predict', title='Predict')

    def get_reference_set(reference_key: str) -> dict:
        reference_set = reference_sets.get(reference_key)
        if reference_set is None:
            raise KeyError(reference_key)
        return reference_set

    def combined_summary(summary: dict | None = None) -> dict:
        gallery_summary = summary or gallery_index.summary()
        dataset_summary = dataset_store.summary()
        not_cat_summary = reference_sets['not_cat']['store'].summary()
        unknown_cat_summary = reference_sets['unknown_cat']['store'].summary()
        actual_total_images = (
            dataset_summary['dataset_images']
            + not_cat_summary['image_count']
            + unknown_cat_summary['image_count']
        )
        indexed_total_images = (
            gallery_summary['gallery_images']
            + gallery_summary['not_cat_images']
            + gallery_summary['unknown_cat_images']
        )

        if actual_total_images == 0:
            index_status = 'empty'
        elif gallery_index.last_build_signature and not gallery_index.is_stale():
            index_status = 'ready'
        else:
            index_status = 'needs_train'

        return {
            **gallery_summary,
            **dataset_summary,
            'not_cat_folder_images': not_cat_summary['image_count'],
            'unknown_cat_folder_images': unknown_cat_summary['image_count'],
            'actual_total_images': actual_total_images,
            'indexed_total_images': indexed_total_images,
            'index_status': index_status,
        }

    train_manager = TrainManager(
        rebuild_fn=gallery_index.rebuild,
        summary_fn=combined_summary,
    )
    app.extensions['train_manager'] = train_manager

    def serialize_cat(cat: dict) -> dict:
        image_names = cat.get('image_names', [])
        images = [
            {
                'name': image_name,
                'url': url_for('api_cat_image', cat_id=cat['id'], image_name=image_name),
            }
            for image_name in image_names
        ]
        return {
            key: value
            for key, value in {
                **cat,
                'cover_image': images[0]['url'] if images else None,
                'images': images,
            }.items()
            if key != 'image_names'
        }

    def serialize_reference_set(reference_key: str, *, limit: int = 5) -> dict:
        reference_set = get_reference_set(reference_key)
        store = reference_set['store']
        sample_images = store.list_images(limit=limit)
        total_images = store.summary()['image_count']
        images = [
            {
                'name': image_path.name,
                'url': url_for('api_reference_image', reference_key=reference_key, image_name=image_path.name),
            }
            for image_path in sample_images
        ]
        return {
            'key': reference_key,
            'label': reference_set['label'],
            'image_count': total_images,
            'hidden_count': max(total_images - len(images), 0),
            'images': images,
        }

    @app.get('/api/status')
    def api_status():
        return jsonify({'status': 'ok', **combined_summary()})

    @app.post('/api/rebuild')
    def api_rebuild():
        summary = gallery_index.rebuild()
        return jsonify({'status': 'ok', **combined_summary(summary)})

    @app.post('/api/train')
    def api_train():
        job, started = train_manager.start()
        return jsonify({'status': 'ok', 'started': started, 'job': job, 'summary': combined_summary()}), (202 if started else 200)

    @app.get('/api/train/status')
    def api_train_status():
        return jsonify({'status': 'ok', 'job': train_manager.snapshot(), 'summary': combined_summary()})

    @app.get('/api/cats')
    def api_cats():
        cats_payload = [serialize_cat(cat) for cat in dataset_store.list_cats()]
        return jsonify({'status': 'ok', 'cats': cats_payload, 'summary': combined_summary()})

    @app.get('/api/reference-sets')
    def api_reference_sets():
        payload = [serialize_reference_set(reference_key) for reference_key in reference_sets]
        return jsonify({'status': 'ok', 'reference_sets': payload, 'summary': combined_summary()})

    @app.get('/api/reference-sets/<reference_key>')
    def api_reference_set(reference_key: str):
        try:
            limit = request.args.get('limit', default=24, type=int) or 24
            return jsonify(
                {
                    'status': 'ok',
                    'reference_set': serialize_reference_set(reference_key, limit=max(limit, 1)),
                    'summary': combined_summary(),
                }
            )
        except KeyError:
            return jsonify({'status': 'error', 'message': 'ไม่พบคลาสอ้างอิง'}), 404

    @app.post('/api/cats')
    def api_create_cat():
        payload = request.form if request.form else (request.get_json(silent=True) or {})
        try:
            cat = dataset_store.create_cat(
                name=payload.get('name', ''),
                owner=payload.get('owner', ''),
                location=payload.get('location', ''),
                files=request.files.getlist('images'),
            )
            return jsonify({'status': 'ok', 'cat': serialize_cat(cat), 'summary': combined_summary()}), 201
        except ValueError as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 400

    @app.put('/api/cats/<int:cat_id>')
    def api_update_cat(cat_id: int):
        payload = request.form if request.form else (request.get_json(silent=True) or {})
        try:
            cat = dataset_store.update_cat(
                cat_id,
                name=payload.get('name', ''),
                owner=payload.get('owner', ''),
                location=payload.get('location', ''),
                files=request.files.getlist('images'),
            )
            return jsonify({'status': 'ok', 'cat': serialize_cat(cat), 'summary': combined_summary()})
        except KeyError:
            return jsonify({'status': 'error', 'message': 'ไม่พบข้อมูลแมว'}), 404
        except ValueError as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 400

    @app.delete('/api/cats/<int:cat_id>')
    def api_delete_cat(cat_id: int):
        try:
            dataset_store.delete_cat(cat_id)
            return jsonify({'status': 'ok', 'summary': combined_summary()})
        except KeyError:
            return jsonify({'status': 'error', 'message': 'ไม่พบข้อมูลแมว'}), 404

    @app.delete('/api/cats/<int:cat_id>/images/<path:image_name>')
    def api_delete_cat_image(cat_id: int, image_name: str):
        try:
            cat = dataset_store.delete_image(cat_id, image_name)
            return jsonify({'status': 'ok', 'cat': serialize_cat(cat), 'summary': combined_summary()})
        except KeyError:
            return jsonify({'status': 'error', 'message': 'ไม่พบข้อมูลแมว'}), 404
        except FileNotFoundError:
            return jsonify({'status': 'error', 'message': 'ไม่พบรูปภาพ'}), 404

    @app.get('/api/cats/<int:cat_id>/images/<path:image_name>')
    def api_cat_image(cat_id: int, image_name: str):
        try:
            return send_file(dataset_store.get_image_path(cat_id, image_name))
        except KeyError:
            return jsonify({'status': 'error', 'message': 'ไม่พบข้อมูลแมว'}), 404
        except FileNotFoundError:
            return jsonify({'status': 'error', 'message': 'ไม่พบรูปภาพ'}), 404

    @app.post('/api/reference-sets/<reference_key>/images')
    def api_reference_upload(reference_key: str):
        try:
            reference_set = get_reference_set(reference_key)
        except KeyError:
            return jsonify({'status': 'error', 'message': 'ไม่พบคลาสอ้างอิง'}), 404

        try:
            reference_set['store'].add_images(request.files.getlist('images'))
            return jsonify(
                {
                    'status': 'ok',
                    'reference_set': serialize_reference_set(reference_key, limit=24),
                    'summary': combined_summary(),
                }
            ), 201
        except ValueError as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 400

    @app.delete('/api/reference-sets/<reference_key>/images/<path:image_name>')
    def api_reference_delete(reference_key: str, image_name: str):
        try:
            reference_set = get_reference_set(reference_key)
            reference_set['store'].delete_image(image_name)
            return jsonify(
                {
                    'status': 'ok',
                    'reference_set': serialize_reference_set(reference_key, limit=24),
                    'summary': combined_summary(),
                }
            )
        except KeyError:
            return jsonify({'status': 'error', 'message': 'ไม่พบคลาสอ้างอิง'}), 404
        except FileNotFoundError:
            return jsonify({'status': 'error', 'message': 'ไม่พบรูปภาพ'}), 404

    @app.get('/api/reference-sets/<reference_key>/images/<path:image_name>')
    def api_reference_image(reference_key: str, image_name: str):
        try:
            reference_set = get_reference_set(reference_key)
            return send_file(reference_set['store'].get_image_path(image_name))
        except KeyError:
            return jsonify({'status': 'error', 'message': 'ไม่พบคลาสอ้างอิง'}), 404
        except FileNotFoundError:
            return jsonify({'status': 'error', 'message': 'ไม่พบรูปภาพ'}), 404

    @app.post('/api/predict')
    def api_predict():
        image = None

        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            image = Image.open(file.stream).convert('RGB')
        else:
            payload = request.get_json(silent=True) or {}
            if payload.get('image_base64'):
                raw = payload['image_base64']
                if ',' in raw:
                    raw = raw.split(',', 1)[1]
                data = base64.b64decode(raw)
                image = Image.open(io.BytesIO(data)).convert('RGB')

        if image is None:
            return jsonify({'status': 'error', 'message': 'Please upload an image file.'}), 400

        try:
            result = pipeline.predict(image)
            return jsonify({'status': 'ok', **result})
        except ValueError as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 400
        except Exception as exc:  # pragma: no cover
            return jsonify({'status': 'error', 'message': f'Unexpected error: {exc}'}), 500

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5001')), debug=True)
