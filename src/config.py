"""Project-wide paths and lightweight model configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.keras"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
EVALUATION_METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

IMAGE_SIZE = (224, 224)
IMAGE_HEIGHT, IMAGE_WIDTH = IMAGE_SIZE
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
RANDOM_SEED = 42
DROPOUT_RATE = 0.2

# image_dataset_from_directory uses alphabetical class order. Keeping the order
# explicit makes the sigmoid output stable: output index 1 is P(Healthy).
CLASS_NAMES = ("Bleached", "Healthy")
POSITIVE_CLASS = CLASS_NAMES[1]
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png"}
)


def model_metadata_path(model_path: str | Path = MODEL_PATH) -> Path:
    """Return the JSON sidecar path used to preserve model class semantics."""

    path = Path(model_path)
    return path.with_name(f"{path.stem}.metadata.json")


def save_model_metadata(
    model_path: str | Path,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    image_size: Sequence[int] = IMAGE_SIZE,
) -> Path:
    """Write preprocessing and label metadata next to a saved Keras model."""

    names = tuple(class_names)
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("Exactly two unique class names are required.")

    size = tuple(int(value) for value in image_size)
    if len(size) != 2 or any(value <= 0 for value in size):
        raise ValueError("image_size must contain two positive integers.")

    metadata = {
        "architecture": "EfficientNetB0",
        "class_names": list(names),
        "positive_class": names[1],
        "image_size": list(size),
        "input_color_mode": "RGB",
        "input_value_range": [0, 255],
        "normalization": "embedded in EfficientNetB0",
    }
    path = model_metadata_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_model_metadata(
    model_path: str | Path = MODEL_PATH,
) -> dict[str, Any] | None:
    """Load a model metadata sidecar, returning ``None`` when it is absent."""

    path = model_metadata_path(model_path)
    if not path.exists():
        return None

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model metadata at {path}: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ValueError(f"Model metadata at {path} must contain a JSON object.")
    return metadata
