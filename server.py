import base64
import json
import math
import os
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from tensorflow import keras


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "server_models"
METADATA_PATH = MODEL_DIR / "models.json"
IMAGE_SIZE = (224, 224)

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")
MODEL_DIR.mkdir(exist_ok=True)
model_cache = {}


@keras.utils.register_keras_serializable()
class L2Normalization(keras.layers.Layer):
  def __init__(self, axis=-1, epsilon=1e-12, **kwargs):
    super().__init__(**kwargs)
    self.axis = axis
    self.epsilon = epsilon

  def call(self, inputs):
    return tf.math.l2_normalize(inputs, axis=self.axis, epsilon=self.epsilon)

  def get_config(self):
    config = super().get_config()
    config.update({"axis": self.axis, "epsilon": self.epsilon})
    return config


@keras.utils.register_keras_serializable()
class L2Norm(L2Normalization):
  pass


@keras.utils.register_keras_serializable()
class DistanceLayer(keras.layers.Layer):
  def __init__(self, mode="l2", epsilon=1e-12, **kwargs):
    super().__init__(**kwargs)
    self.mode = mode
    self.epsilon = epsilon

  def call(self, inputs):
    left, right = inputs
    diff = left - right
    if self.mode in ("l1", "abs", "manhattan"):
      return tf.abs(diff)
    return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1, keepdims=True) + self.epsilon)

  def get_config(self):
    config = super().get_config()
    config.update({"mode": self.mode, "epsilon": self.epsilon})
    return config


CUSTOM_OBJECTS = {
  "L2Normalization": L2Normalization,
  "L2Normalize": L2Normalization,
  "L2Norm": L2Norm,
  "l2_norm": tf.math.l2_normalize,
  "l2_normalize": tf.math.l2_normalize,
  "DistanceLayer": DistanceLayer,
  "L1Distance": DistanceLayer,
  "L2Distance": DistanceLayer,
  "EuclideanDistance": DistanceLayer,
  "ManhattanDistance": DistanceLayer,
}


def load_keras_model(path):
  with keras.utils.custom_object_scope(CUSTOM_OBJECTS):
    try:
      return keras.models.load_model(
        path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
        safe_mode=False,
      )
    except TypeError:
      return keras.models.load_model(
        path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
      )


def read_models():
  if not METADATA_PATH.exists():
    return []
  try:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
    return []


def write_models(models):
  METADATA_PATH.write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")


def public_model_meta(model):
  return {
    "id": model["id"],
    "name": model["name"],
    "size": model["size"],
    "uploadedAt": model["uploadedAt"],
    "active": model.get("active", False),
    "backend": True,
    "format": model.get("format", "keras-h5"),
    "outputMode": model.get("outputMode", "auto"),
  }


def load_backend_model(model_id):
  if model_id in model_cache:
    return model_cache[model_id]
  model_meta = next((m for m in read_models() if m["id"] == model_id), None)
  if not model_meta:
    raise ValueError("Model not found")
  model = load_keras_model(model_meta["path"])
  model_cache[model_id] = model
  return model


def infer_output_mode(model):
  if len(model.inputs) < 2:
    return "embedding"
  layer = model.layers[-1]
  layer_name = layer.name.lower()
  class_name = layer.__class__.__name__.lower()
  activation = getattr(layer, "activation", None)
  activation_name = getattr(activation, "__name__", "").lower()
  descriptor = f"{layer_name} {class_name}"
  if activation_name in ("sigmoid", "softmax"):
    return "score"
  if any(token in descriptor for token in ("distance", "euclidean", "manhattan", "lambda", "l1", "l2")):
    return "distance"
  if activation_name in ("linear", ""):
    return "distance"
  return "score"


def parse_data_url(src):
  if "," in src:
    src = src.split(",", 1)[1]
  raw = base64.b64decode(src)
  image = Image.open(BytesIO(raw)).convert("RGB")
  image = image.resize(IMAGE_SIZE)
  arr = np.asarray(image, dtype=np.float32) / 255.0
  return np.expand_dims(arr, axis=0)


def normalize_score(value, output_mode="auto"):
  value = float(value)
  if not math.isfinite(value):
    return 0.0
  if output_mode == "distance":
    return round(1.0 / (1.0 + max(0.0, value)), 4)
  if 0.0 <= value <= 1.0:
    return round(value, 4)
  return round(1.0 / (1.0 + math.exp(-value)), 4)


def cosine_score(left, right):
  left = np.asarray(left).reshape(-1)
  right = np.asarray(right).reshape(-1)
  denom = np.linalg.norm(left) * np.linalg.norm(right)
  if denom == 0:
    return 0.0
  cosine = float(np.dot(left, right) / denom)
  return round(float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)), 4)


def raw_cosine(left, right):
  left = np.asarray(left).reshape(-1)
  right = np.asarray(right).reshape(-1)
  denom = np.linalg.norm(left) * np.linalg.norm(right)
  if denom == 0:
    return 0.0
  return round(float(np.dot(left, right) / denom), 4)


@app.get("/")
def index():
  return send_from_directory(ROOT, "index.html")


@app.get("/api/health")
def health():
  return jsonify({"ok": True, "backend": "keras-h5"})


@app.get("/api/models")
def list_models():
  return jsonify({"models": [public_model_meta(model) for model in read_models()]})


@app.post("/api/models")
def upload_model():
  file = request.files.get("model")
  if not file or not file.filename.lower().endswith((".h5", ".keras")):
    return jsonify({"error": "Upload a .h5 or .keras model file"}), 400

  models = read_models()
  model_id = str(uuid.uuid4())
  suffix = Path(file.filename).suffix or ".h5"
  path = MODEL_DIR / f"{model_id}{suffix}"
  file.save(path)

  try:
    model = load_keras_model(path)
    model_cache[model_id] = model
  except Exception as exc:
    path.unlink(missing_ok=True)
    return jsonify({"error": f"Cannot load model: {exc}"}), 400

  for model in models:
    model["active"] = False
  meta = {
    "id": model_id,
    "name": file.filename,
    "size": f"{path.stat().st_size / 1024 / 1024:.2f} MB",
    "uploadedAt": request.form.get("uploadedAt") or "",
    "active": True,
    "format": "keras-h5",
    "outputMode": infer_output_mode(model),
    "path": str(path),
  }
  models.insert(0, meta)
  write_models(models)
  return jsonify({"model": public_model_meta(meta)})


@app.post("/api/models/<model_id>/activate")
def activate_model(model_id):
  models = read_models()
  found = False
  for model in models:
    model["active"] = model["id"] == model_id
    found = found or model["active"]
  if not found:
    return jsonify({"error": "Model not found"}), 404
  try:
    model = load_backend_model(model_id)
  except Exception as exc:
    return jsonify({"error": f"Cannot load model: {exc}"}), 400
  for model_meta in models:
    if model_meta["id"] == model_id:
      model_meta["outputMode"] = infer_output_mode(model)
  active_meta = next((model_meta for model_meta in models if model_meta["id"] == model_id), None)
  write_models(models)
  return jsonify({"ok": True, "model": public_model_meta(active_meta)})


@app.delete("/api/models")
def clear_models():
  for model in read_models():
    Path(model["path"]).unlink(missing_ok=True)
  model_cache.clear()
  write_models([])
  return jsonify({"ok": True})


@app.post("/api/compare")
def compare():
  payload = request.get_json(force=True)
  model_id = payload.get("modelId")
  left_src = payload.get("leftSrc")
  right_src = payload.get("rightSrc")
  if not model_id or not left_src or not right_src:
    return jsonify({"error": "modelId, leftSrc, and rightSrc are required"}), 400

  try:
    model = load_backend_model(model_id)
    output_mode = infer_output_mode(model)
    models = read_models()
    for model_meta in models:
      if model_meta["id"] == model_id:
        model_meta["outputMode"] = output_mode
    write_models(models)

    left = parse_data_url(left_src)
    right = parse_data_url(right_src)
    if len(model.inputs) >= 2:
      output = model.predict([left, right], verbose=0)
      raw_value = float(np.asarray(output).reshape(-1)[0])
      score = normalize_score(raw_value, output_mode)
      mode = "pair"
      display_label = "Distance" if output_mode == "distance" else "Similarity Score"
      display_value = raw_value
    else:
      left_embedding = model.predict(left, verbose=0)
      right_embedding = model.predict(right, verbose=0)
      raw_value = None
      score = cosine_score(left_embedding, right_embedding)
      embedding_cosine = raw_cosine(left_embedding, right_embedding)
      mode = "encoder"
      output_mode = "embedding"
      display_label = "Embedding Similarity"
      display_value = score
    return jsonify({
      "score": score,
      "mode": mode,
      "raw": raw_value,
      "rawCosine": embedding_cosine if mode == "encoder" else None,
      "outputMode": output_mode,
      "displayLabel": display_label,
      "displayValue": display_value,
    })
  except Exception as exc:
    return jsonify({"error": str(exc)}), 400


@app.get("/<path:path>")
def static_files(path):
  return send_from_directory(ROOT, path)


if __name__ == "__main__":
  app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5500")), debug=True)
