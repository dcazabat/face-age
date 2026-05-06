import argparse
import base64
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, url_for
from logging.handlers import RotatingFileHandler

from main import analyze_frame, ensure_models, load_nets, resolve_haar_cascade_path


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("face_age_runtime")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            logs_dir / "face-age.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    return logger


class FaceService:
    def __init__(self, models_dir: str) -> None:
        self.lock = threading.Lock()

        model_paths = ensure_models(Path(models_dir))
        self.age_net, self.gender_net, self.emotion_net = load_nets(model_paths)
        self.face_cascade = cv2.CascadeClassifier(resolve_haar_cascade_path())
        if self.face_cascade.empty():
            raise RuntimeError("No se pudo cargar haarcascade_frontalface_default.xml.")

    @staticmethod
    def _decode_data_url(image_data_url: str) -> np.ndarray:
        if "," not in image_data_url:
            raise RuntimeError("Formato de imagen inválido.")
        _, encoded = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        frame_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("No se pudo decodificar la imagen enviada.")
        return frame

    def analyze_image(self, image_data_url: str) -> dict[str, Any]:
        frame = self._decode_data_url(image_data_url)
        with self.lock:
            predictions = analyze_frame(
                frame,
                self.face_cascade,
                self.age_net,
                self.gender_net,
                self.emotion_net,
            )

        return {
            "ok": True,
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
            "predictions": [
                {
                    "x": p.x,
                    "y": p.y,
                    "w": p.w,
                    "h": p.h,
                    "age": p.age,
                    "gender": p.gender,
                    "emotion": p.emotion,
                    "confidence": round(p.confidence, 1),
                }
                for p in predictions
            ],
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frontend Flask para analisis facial.")
    parser.add_argument("--host", default="127.0.0.1", help="Host de Flask.")
    parser.add_argument("--port", type=int, default=5000, help="Puerto de Flask.")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directorio para descargar y guardar modelos.",
    )
    return parser.parse_args()


def create_app(service: FaceService) -> Flask:
    app = Flask(__name__)
    logger = configure_logger()

    @app.get("/")
    def index() -> str:
        analyze_url = os.environ.get("FACE_AGE_ANALYZE_URL") or url_for("analyze", _external=True)
        client_log_url = os.environ.get("FACE_AGE_CLIENT_LOG_URL") or url_for("client_log", _external=True)
        return render_template(
            "index.html",
            analyze_url=analyze_url,
            client_log_url=client_log_url,
        )

    @app.post("/client-log")
    def client_log() -> Response:
        payload = request.get_json(silent=True) or {}
        event = str(payload.get("event", "unknown")).strip() or "unknown"
        message = str(payload.get("message", "")).strip()
        details = payload.get("details")
        logger.info(
            "client_event=%s ip=%s message=%s details=%s",
            event,
            request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown",
            message,
            json.dumps(details, ensure_ascii=False, default=str),
        )
        return jsonify({"ok": True})

    @app.post("/analyze")
    def analyze() -> Response:
        payload = request.get_json(silent=True) or {}
        image = payload.get("image")
        if not image:
            logger.warning(
                "analyze_failed ip=%s reason=missing_image",
                request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown",
            )
            return jsonify({"ok": False, "error": "Falta la imagen."}), 400
        try:
            result = service.analyze_image(str(image))
            return jsonify(result)
        except RuntimeError as err:
            logger.warning(
                "analyze_failed ip=%s error=%s",
                request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown",
                err,
            )
            return jsonify({"ok": False, "error": str(err)}), 400

    return app


def main() -> None:
    args = parse_args()
    service = FaceService(models_dir=args.models_dir)
    app = create_app(service)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
