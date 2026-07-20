"""Single-image inference for the saved coral health classifier."""

from __future__ import annotations

import argparse
import json
from os import PathLike
from pathlib import Path
from typing import BinaryIO, Sequence

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

try:
    from .config import (
        CLASS_NAMES,
        IMAGE_SIZE,
        MODEL_PATH,
        load_model_metadata,
    )
except ImportError:  # Support direct execution with ``python src/predict.py``.
    from config import (  # type: ignore
        CLASS_NAMES,
        IMAGE_SIZE,
        MODEL_PATH,
        load_model_metadata,
    )


ImageInput = (
    str
    | PathLike[str]
    | bytes
    | bytearray
    | BinaryIO
    | Image.Image
    | np.ndarray
)


def load_trained_model(
    model_path: str | Path = MODEL_PATH,
) -> tf.keras.Model:
    """Load a saved Keras model for inference without optimizer state."""

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {path}. Run training first."
        )
    return tf.keras.models.load_model(path, compile=False)


def _decode_encoded_image(encoded: bytes, source: str) -> np.ndarray:
    if not encoded:
        raise ValueError(f"No image bytes were provided by {source}.")
    buffer = np.frombuffer(encoded, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError(f"OpenCV could not decode an image from {source}.")

    if decoded.ndim == 2:
        return cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    if decoded.ndim != 3:
        raise ValueError(f"Unsupported decoded image shape: {decoded.shape!r}")
    if decoded.shape[2] == 1:
        return cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    if decoded.shape[2] == 3:
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if decoded.shape[2] == 4:
        return cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Unsupported decoded channel count: {decoded.shape[2]}")


def _path_to_rgb(path_value: str | PathLike[str]) -> np.ndarray:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    # np.fromfile + imdecode handles Unicode Windows paths more reliably than
    # cv2.imread while still using OpenCV for decoding.
    try:
        encoded = np.fromfile(path, dtype=np.uint8).tobytes()
    except OSError as exc:
        raise ValueError(f"Could not read image file {path}: {exc}") from exc
    return _decode_encoded_image(encoded, str(path))


def _array_to_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.ndim == 3 and array.shape[2] == 4:
        # Arrays supplied by callers are treated as RGB/RGBA, unlike OpenCV's
        # encoded-image decoder which returns BGR/BGRA.
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            "Image arrays must have shape (height, width), (height, width, 1), "
            "(height, width, 3), or (height, width, 4)."
        )

    if array.size == 0:
        raise ValueError("Image array is empty.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Image array values must be numeric.")
    array = array.astype(np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("Image array contains NaN or infinite values.")
    if np.min(array) < 0:
        raise ValueError("Image array values cannot be negative.")
    if np.max(array) <= 1.0:
        array *= 255.0
    elif np.max(array) > 255.0:
        # Support higher bit-depth decoded images without clipping all detail.
        array *= 255.0 / np.max(array)
    return np.clip(array, 0.0, 255.0)


def _read_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, Image.Image):
        corrected = ImageOps.exif_transpose(image).convert("RGB")
        return _array_to_rgb(np.asarray(corrected))
    if isinstance(image, np.ndarray):
        return _array_to_rgb(image)
    if isinstance(image, (str, PathLike)):
        return _array_to_rgb(_path_to_rgb(image))
    if isinstance(image, (bytes, bytearray)):
        return _array_to_rgb(_decode_encoded_image(bytes(image), "memory"))

    if hasattr(image, "getvalue"):
        encoded = image.getvalue()
    elif hasattr(image, "read"):
        position: int | None = None
        if hasattr(image, "tell"):
            try:
                position = image.tell()
            except (OSError, ValueError):
                position = None
        encoded = image.read()
        if position is not None and hasattr(image, "seek"):
            try:
                image.seek(position)
            except (OSError, ValueError):
                pass
    else:
        raise TypeError(
            "image must be a path, encoded bytes, file-like object, PIL Image, "
            "or NumPy array."
        )

    if not isinstance(encoded, (bytes, bytearray)):
        raise TypeError("The file-like image object must return bytes.")
    return _array_to_rgb(_decode_encoded_image(bytes(encoded), "file-like object"))


def preprocess_image(
    image: ImageInput,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Decode, convert to RGB, resize, and add a batch dimension.

    Returned values are float32 pixels in [0, 255]. EfficientNetB0 performs its
    own normalization inside the saved model.
    """

    size = tuple(int(value) for value in image_size)
    if len(size) != 2 or any(value <= 0 for value in size):
        raise ValueError("image_size must contain two positive integers.")
    rgb = _read_image(image)
    # Match image_dataset_from_directory(..., interpolation="bilinear") so
    # command-line and Streamlit inference see the same resize transform used
    # by training and evaluation. OpenCV remains responsible for robust image
    # decoding and color conversion above.
    resized = tf.image.resize(
        rgb,
        size,
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    return np.ascontiguousarray(batch)


def _validated_class_names(
    model_path: str | Path,
    class_names: Sequence[str],
) -> tuple[str, str]:
    names = tuple(str(name) for name in class_names)
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("Exactly two unique class names are required.")

    metadata = load_model_metadata(model_path)
    if metadata is not None:
        stored = metadata.get("class_names")
        if not isinstance(stored, list) or len(stored) != 2:
            raise ValueError("Model metadata must contain exactly two class_names.")
        stored_names = tuple(str(name) for name in stored)
        if stored_names != names:
            raise ValueError(
                "Requested class order does not match the saved model metadata: "
                f"{names!r} != {stored_names!r}."
            )
    return names[0], names[1]


def _healthy_probability(
    image: ImageInput,
    model: tf.keras.Model,
    *,
    image_size: tuple[int, int],
) -> float:
    batch = preprocess_image(image, image_size=image_size)
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, tuple) and len(input_shape) == 4:
        expected = input_shape[1:3]
        if all(dimension is not None for dimension in expected):
            actual = tuple(batch.shape[1:3])
            if tuple(expected) != actual:
                raise ValueError(
                    f"Model expects images sized {tuple(expected)}, got {actual}."
                )

    raw_prediction = model.predict(batch, verbose=0)
    if isinstance(raw_prediction, (list, tuple, dict)):
        raise ValueError("Expected a single sigmoid model output.")
    flattened = np.asarray(raw_prediction).reshape(-1)
    if flattened.size != 1:
        raise ValueError(
            f"Expected one sigmoid probability, received shape "
            f"{np.asarray(raw_prediction).shape}."
        )
    probability = float(flattened[0])
    if not np.isfinite(probability) or not -1e-6 <= probability <= 1 + 1e-6:
        raise ValueError("Model prediction is not a probability in [0, 1].")
    return float(np.clip(probability, 0.0, 1.0))


def predict_probabilities(
    image: ImageInput,
    model: tf.keras.Model | None = None,
    model_path: str | Path = MODEL_PATH,
    class_names: Sequence[str] = CLASS_NAMES,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> dict[str, float]:
    """Return probabilities for both classes in the preserved class order."""

    names = _validated_class_names(model_path, class_names)
    inference_model = model if model is not None else load_trained_model(model_path)
    healthy_probability = _healthy_probability(
        image,
        inference_model,
        image_size=image_size,
    )
    return {
        names[0]: 1.0 - healthy_probability,
        names[1]: healthy_probability,
    }


def predict_image(
    image: ImageInput,
    model: tf.keras.Model | None = None,
    model_path: str | Path = MODEL_PATH,
    class_names: Sequence[str] = CLASS_NAMES,
    image_size: tuple[int, int] = IMAGE_SIZE,
    threshold: float = 0.5,
) -> tuple[str, float]:
    """Predict one image and return ``(class_label, confidence_0_to_1)``."""

    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1.")
    names = _validated_class_names(model_path, class_names)
    inference_model = model if model is not None else load_trained_model(model_path)
    healthy_probability = _healthy_probability(
        image,
        inference_model,
        image_size=image_size,
    )
    if healthy_probability >= threshold:
        return names[1], healthy_probability
    return names[0], 1.0 - healthy_probability


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify one coral reef image as Healthy or Bleached.",
    )
    parser.add_argument("image", type=Path, help="Path to an underwater image.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of human-readable text.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    label, confidence = predict_image(
        args.image,
        model_path=args.model_path,
        threshold=args.threshold,
    )
    if args.json:
        print(json.dumps({"label": label, "confidence": confidence}, indent=2))
    else:
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
