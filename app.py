from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

from services.gallery import GalleryIndex
from services.pipeline import CatIdentityPipeline, PipelineConfig

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
INDEX_DIR = DATA_DIR / 'index'
GALLERY_DIR = DATA_DIR / 'gallery'
NOT_CAT_DIR = DATA_DIR / 'not_cat'
UNKNOWN_CAT_DIR = DATA_DIR / 'unknown_cat'


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY='dev',
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        DATA_DIR=DATA_DIR,
        INDEX_DIR=INDEX_DIR,
        GALLERY_DIR=GALLERY_DIR,
        NOT_CAT_DIR=NOT_CAT_DIR,
        UNKNOWN_CAT_DIR=UNKNOWN_CAT_DIR,
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
    gallery_index = GalleryIndex(cfg)
    pipeline = CatIdentityPipeline(cfg, gallery_index)
    app.extensions['gallery_index'] = gallery_index
    app.extensions['pipeline'] = pipeline

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/status')
    def api_status():
        summary = gallery_index.summary()
        return jsonify({'status': 'ok', **summary})

    @app.post('/api/rebuild')
    def api_rebuild():
        summary = gallery_index.rebuild()
        return jsonify({'status': 'ok', **summary})

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
    app.run(debug=True, port=5001)
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=True)
