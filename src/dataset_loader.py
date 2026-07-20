"""TensorFlow input pipelines for the train, validation, and test splits."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import tensorflow as tf

try:
    from .config import (
        BATCH_SIZE,
        CLASS_NAMES,
        DATASET_DIR,
        IMAGE_SIZE,
        RANDOM_SEED,
        SUPPORTED_IMAGE_EXTENSIONS,
    )
except ImportError:  # Support ``python src/dataset_loader.py`` style imports.
    from config import (  # type: ignore
        BATCH_SIZE,
        CLASS_NAMES,
        DATASET_DIR,
        IMAGE_SIZE,
        RANDOM_SEED,
        SUPPORTED_IMAGE_EXTENSIONS,
    )


SPLIT_NAMES = ("train", "valid", "test")


def _validate_class_names(class_names: Sequence[str]) -> tuple[str, str]:
    names = tuple(str(name) for name in class_names)
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("This binary classifier requires two unique class names.")
    return names[0], names[1]


def _count_images(directory: Path) -> int:
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def validate_dataset_structure(
    dataset_dir: str | Path = DATASET_DIR,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    split_names: Sequence[str] = SPLIT_NAMES,
) -> dict[str, dict[str, int]]:
    """Validate all expected folders and return image counts by split/class."""

    root = Path(dataset_dir)
    names = _validate_class_names(class_names)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. Expected train/valid/test "
            "folders inside it."
        )

    counts: dict[str, dict[str, int]] = {}
    errors: list[str] = []
    for split_name in split_names:
        split_dir = root / split_name
        if not split_dir.is_dir():
            errors.append(f"missing split directory: {split_dir}")
            continue

        visible_directories = {
            path.name
            for path in split_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        missing_classes = set(names) - visible_directories
        unexpected_classes = visible_directories - set(names)
        if missing_classes:
            errors.append(
                f"{split_dir} is missing class folder(s): "
                f"{', '.join(sorted(missing_classes))}"
            )
        if unexpected_classes:
            errors.append(
                f"{split_dir} has unexpected class folder(s): "
                f"{', '.join(sorted(unexpected_classes))}"
            )

        split_counts: dict[str, int] = {}
        for class_name in names:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            image_count = _count_images(class_dir)
            split_counts[class_name] = image_count
            if image_count == 0:
                errors.append(f"no supported images found in {class_dir}")
        counts[str(split_name)] = split_counts

    if errors:
        details = "\n - ".join(errors)
        raise ValueError(f"Invalid dataset structure:\n - {details}")
    return counts


def _prepare_efficientnet_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Cast images and invoke the documented EfficientNet preprocessing API.

    Modern Keras EfficientNet models embed their normalization layer, so
    ``preprocess_input`` intentionally preserves the 0-255 pixel range. Dividing
    here by 255 would normalize the same data twice and harm transfer learning.
    """

    images = tf.cast(images, tf.float32)
    images = tf.keras.applications.efficientnet.preprocess_input(images)
    return images, labels


def load_dataset(
    directory: str | Path,
    *,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    class_names: Sequence[str] = CLASS_NAMES,
    shuffle: bool = False,
    seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    """Load one directory split as a prefetched binary dataset."""

    path = Path(directory)
    names = _validate_class_names(class_names)
    size = tuple(int(value) for value in image_size)
    if len(size) != 2 or any(value <= 0 for value in size):
        raise ValueError("image_size must contain two positive integers.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {path}")

    for class_name in names:
        class_dir = path / class_name
        if not class_dir.is_dir():
            raise ValueError(f"Missing class directory: {class_dir}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="binary",
        class_names=list(names),
        color_mode="rgb",
        batch_size=batch_size,
        image_size=size,
        shuffle=shuffle,
        seed=seed,
        interpolation="bilinear",
    )
    dataset = dataset.map(
        _prepare_efficientnet_batch,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def load_datasets(
    dataset_dir: str | Path = DATASET_DIR,
    *,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    class_names: Sequence[str] = CLASS_NAMES,
    seed: int = RANDOM_SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train, validation, and test splits with a stable class mapping."""

    root = Path(dataset_dir)
    validate_dataset_structure(root, class_names=class_names)

    train_dataset = load_dataset(
        root / "train",
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names,
        shuffle=True,
        seed=seed,
    )
    validation_dataset = load_dataset(
        root / "valid",
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names,
        shuffle=False,
        seed=seed,
    )
    test_dataset = load_dataset(
        root / "test",
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names,
        shuffle=False,
        seed=seed,
    )
    return train_dataset, validation_dataset, test_dataset

