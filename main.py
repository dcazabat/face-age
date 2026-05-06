import argparse
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Evita warnings de Qt en ruedas de OpenCV que no incluyen fuentes embebidas.
for font_dir in (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/dejavu",
):
    if Path(font_dir).exists():
        os.environ.setdefault("QT_QPA_FONTDIR", font_dir)
        break

import cv2
import numpy as np


@dataclass
class FacePrediction:
    x: int
    y: int
    w: int
    h: int
    age: int
    gender: str
    emotion: str
    confidence: float


AGE_BUCKETS = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]

GENDER_BUCKETS = ["Male", "Female"]

EMOTION_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

MODEL_URLS = {
    "age_prototxt": "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/deploy_age2.prototxt",
    "age_model": "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel",
    "gender_prototxt": "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/deploy_gender2.prototxt",
    "gender_model": "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel",
    "emotion_model": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
}

AGE_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


def resolve_haar_cascade_path() -> str:
    candidates: list[Path] = []
    cv2_data = getattr(cv2, "data", None)
    if cv2_data is not None and getattr(cv2_data, "haarcascades", None):
        candidates.append(
            Path(cv2_data.haarcascades) / "haarcascade_frontalface_default.xml"
        )

    candidates.extend(
        [
            Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
            Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise RuntimeError("No se encontró haarcascade_frontalface_default.xml.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconocimiento por camara: edad aproximada, genero y emocion."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indice de la camara a usar (default: 0).",
    )
    parser.add_argument(
        "--analysis-interval",
        type=int,
        default=5,
        help="Analizar cada N frames para mejorar rendimiento (default: 5).",
    )
    parser.add_argument(
        "--detector-backend",
        type=str,
        default="haar",
        choices=["haar"],
        help="Detector de rostro (actualmente: haar).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directorio para descargar y guardar los modelos.",
    )
    return parser.parse_args()


def ensure_models(models_dir: Path) -> dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "age_prototxt": models_dir / "deploy_age2.prototxt",
        "age_model": models_dir / "age_net.caffemodel",
        "gender_prototxt": models_dir / "deploy_gender2.prototxt",
        "gender_model": models_dir / "gender_net.caffemodel",
        "emotion_model": models_dir / "emotion-ferplus-8.onnx",
    }

    for key, path in paths.items():
        if path.exists() and path.stat().st_size > 0:
            continue
        url = MODEL_URLS[key]
        print(f"Descargando modelo: {path.name}")
        try:
            urllib.request.urlretrieve(url, path)
        except urllib.error.URLError as err:
            raise RuntimeError(f"No se pudo descargar {path.name} desde {url}: {err}") from err

    return paths


def load_nets(model_paths: dict[str, Path]) -> tuple[cv2.dnn.Net, cv2.dnn.Net, cv2.dnn.Net]:
    age_net = cv2.dnn.readNet(
        str(model_paths["age_model"]), str(model_paths["age_prototxt"])
    )
    gender_net = cv2.dnn.readNet(
        str(model_paths["gender_model"]), str(model_paths["gender_prototxt"])
    )
    emotion_net = cv2.dnn.readNetFromONNX(str(model_paths["emotion_model"]))
    return age_net, gender_net, emotion_net


def detect_faces(frame: np.ndarray, face_cascade: cv2.CascadeClassifier) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def classify_face(
    face_bgr: np.ndarray, age_net: cv2.dnn.Net, gender_net: cv2.dnn.Net, emotion_net: cv2.dnn.Net
) -> tuple[int, str, str, float]:
    face_227 = cv2.resize(face_bgr, (227, 227))
    blob = cv2.dnn.blobFromImage(
        face_227,
        1.0,
        (227, 227),
        AGE_MEAN_VALUES,
        swapRB=False,
    )

    age_net.setInput(blob)
    age_pred = age_net.forward()[0]
    age_idx = int(np.argmax(age_pred))

    gender_net.setInput(blob)
    gender_pred = gender_net.forward()[0]
    gender_idx = int(np.argmax(gender_pred))

    gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (64, 64))
    emotion_blob = cv2.dnn.blobFromImage(
        gray_face,
        1.0 / 255.0,
        (64, 64),
        (0, 0, 0),
        swapRB=False,
    )
    emotion_net.setInput(emotion_blob)
    emotion_pred = emotion_net.forward()[0]
    emotion_idx = int(np.argmax(emotion_pred))

    emotion_scores = np.exp(emotion_pred - np.max(emotion_pred))
    emotion_scores = emotion_scores / np.sum(emotion_scores)
    confidence = float(emotion_scores[emotion_idx]) * 100.0

    age_label = AGE_BUCKETS[age_idx]
    age_value = int(age_label.strip("()").split("-")[0])
    gender = GENDER_BUCKETS[gender_idx]
    emotion = EMOTION_LABELS[emotion_idx]

    return age_value, gender, emotion, confidence


def analyze_frame(
    frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    age_net: cv2.dnn.Net,
    gender_net: cv2.dnn.Net,
    emotion_net: cv2.dnn.Net,
) -> list[FacePrediction]:
    predictions: list[FacePrediction] = []
    for (x, y, w, h) in detect_faces(frame, face_cascade):
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(frame.shape[1], x + w)
        y1 = min(frame.shape[0], y + h)
        face = frame[y0:y1, x0:x1]
        if face.size == 0:
            continue

        age, gender, emotion, confidence = classify_face(
            face, age_net, gender_net, emotion_net
        )
        predictions.append(
            FacePrediction(
                x=x0,
                y=y0,
                w=x1 - x0,
                h=y1 - y0,
                age=age,
                gender=gender,
                emotion=emotion,
                confidence=confidence,
            )
        )
    return predictions


def draw_predictions(frame: np.ndarray, predictions: list[FacePrediction]) -> None:
    for p in predictions:
        x1, y1 = max(0, p.x), max(0, p.y)
        x2, y2 = max(0, p.x + p.w), max(0, p.y + p.h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)

        text = f"Edad: ~{p.age}+ | Genero: {p.gender} | Emocion: {p.emotion} ({p.confidence:.1f}%)"
        text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
        cv2.putText(
            frame,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    model_paths = ensure_models(Path(args.models_dir))
    age_net, gender_net, emotion_net = load_nets(model_paths)
    face_cascade = cv2.CascadeClassifier(resolve_haar_cascade_path())
    if face_cascade.empty():
        raise RuntimeError("No se pudo cargar haarcascade_frontalface_default.xml.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la camara con indice {args.camera_index}. "
            "Probá con --camera-index 1 o revisá permisos."
        )

    print("Iniciando analisis en tiempo real. Presiona 'q' para salir.")

    frame_number = 0
    latest_predictions: list[FacePrediction] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("No se pudo leer un frame de la camara.")

        frame_number += 1
        if frame_number % args.analysis_interval == 0:
            latest_predictions = analyze_frame(
                frame,
                face_cascade,
                age_net,
                gender_net,
                emotion_net,
            )

        draw_predictions(frame, latest_predictions)
        cv2.imshow("Face Age Gender Emotion", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
