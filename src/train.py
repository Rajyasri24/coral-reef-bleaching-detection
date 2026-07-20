"""Train an EfficientNetB0 coral health classifier on a local CPU or GPU."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import tensorflow as tf

try:
    from .config import (
        BATCH_SIZE,
        CLASS_NAMES,
        DATASET_DIR,
        DROPOUT_RATE,
        EPOCHS,
        FINE_TUNE_LEARNING_RATE,
        IMAGE_SIZE,
        LEARNING_RATE,
        MODEL_PATH,
        RANDOM_SEED,
        load_model_metadata,
        save_model_metadata,
    )
    from .dataset_loader import load_datasets
except ImportError:  # Support direct execution with ``python src/train.py``.
    from config import (  # type: ignore
        BATCH_SIZE,
        CLASS_NAMES,
        DATASET_DIR,
        DROPOUT_RATE,
        EPOCHS,
        FINE_TUNE_LEARNING_RATE,
        IMAGE_SIZE,
        LEARNING_RATE,
        MODEL_PATH,
        RANDOM_SEED,
        load_model_metadata,
        save_model_metadata,
    )
    from dataset_loader import load_datasets  # type: ignore


def _compile_model(
    model: tf.keras.Model,
    learning_rate: float,
) -> None:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def build_model(
    image_size: tuple[int, int] = IMAGE_SIZE,
    learning_rate: float = LEARNING_RATE,
    *,
    weights: str | None = "imagenet",
    dropout_rate: float = DROPOUT_RATE,
) -> tf.keras.Model:
    """Build a frozen EfficientNetB0 backbone with a binary classifier head."""

    size = tuple(int(value) for value in image_size)
    if len(size) != 2 or any(value <= 0 for value in size):
        raise ValueError("image_size must contain two positive integers.")
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be in the range [0, 1).")
    if isinstance(weights, str) and weights.lower() == "none":
        weights = None

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(*size, 3),
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*size, 3), name="image")
    x = augmentation(inputs)
    # EfficientNetB0 contains its own Rescaling normalization layer and expects
    # RGB values in [0, 255]. The backbone runs in inference mode while frozen.
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="healthy_probability",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="coral_classifier")
    _compile_model(model, learning_rate)
    return model


class _CheckpointAwareEarlyStopping(tf.keras.callbacks.EarlyStopping):
    """Treat a loaded checkpoint as the best pre-existing epoch."""

    def __init__(
        self,
        *,
        initial_best: float | None = None,
        initial_epoch: int = 0,
        **kwargs: object,
    ) -> None:
        super().__init__(baseline=initial_best, **kwargs)
        self._initial_best = initial_best
        self._initial_epoch = initial_epoch

    def on_train_begin(self, logs: dict | None = None) -> None:
        super().on_train_begin(logs)
        if self._initial_best is None:
            return

        self.best = self._initial_best
        if self.restore_best_weights:
            self.best_weights = self.model.get_weights()
            self.best_epoch = max(0, self._initial_epoch - 1)


def _training_callbacks(
    model_path: Path,
    *,
    initial_best: float | None = None,
    initial_epoch: int = 0,
) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            initial_value_threshold=initial_best,
            verbose=1,
        ),
        _CheckpointAwareEarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            initial_best=initial_best,
            initial_epoch=initial_epoch,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def _validate_resume_metadata(
    model_path: Path,
    image_size: tuple[int, int],
) -> None:
    metadata = load_model_metadata(model_path)
    if metadata is None:
        return

    if metadata.get("architecture") != "EfficientNetB0":
        raise ValueError(
            "Resume checkpoint metadata does not describe EfficientNetB0."
        )
    if tuple(metadata.get("class_names", ())) != tuple(CLASS_NAMES):
        raise ValueError(
            "Resume checkpoint class names do not match the configured class order."
        )
    if tuple(metadata.get("image_size", ())) != tuple(image_size):
        raise ValueError(
            "Resume checkpoint image size does not match the requested image size."
        )


def _validate_resume_model(
    model: tf.keras.Model,
    image_size: tuple[int, int],
) -> tf.keras.Model:
    expected_input_shape = (None, *tuple(image_size), 3)
    if tuple(model.input_shape) != expected_input_shape:
        raise ValueError(
            "Resume checkpoint input shape does not match the requested image "
            f"size: expected {expected_input_shape}, got {model.input_shape}."
        )
    if tuple(model.output_shape) != (None, 1):
        raise ValueError(
            "Resume checkpoint must have one sigmoid classifier output; "
            f"got {model.output_shape}."
        )

    try:
        backbone = model.get_layer("efficientnetb0")
        augmentation = model.get_layer("data_augmentation")
        pooling = model.get_layer("global_average_pooling")
        dropout = model.get_layer("classifier_dropout")
        output = model.get_layer("healthy_probability")
        backbone.get_layer("top_conv")
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "Resume checkpoint architecture is not the expected EfficientNetB0 "
            "coral classifier."
        ) from exc

    if not isinstance(backbone, tf.keras.Model):
        raise ValueError("Resume checkpoint has an invalid EfficientNetB0 backbone.")
    if model.name != "coral_classifier":
        raise ValueError("Resume checkpoint has an unexpected model architecture.")
    if not isinstance(augmentation, tf.keras.Sequential):
        raise ValueError("Resume checkpoint has an invalid augmentation pipeline.")
    if not isinstance(pooling, tf.keras.layers.GlobalAveragePooling2D):
        raise ValueError("Resume checkpoint has an invalid classifier head.")
    if not isinstance(dropout, tf.keras.layers.Dropout):
        raise ValueError("Resume checkpoint has an invalid classifier head.")
    if not isinstance(output, tf.keras.layers.Dense) or output.units != 1:
        raise ValueError("Resume checkpoint has an invalid classifier output.")
    if getattr(output.activation, "__name__", None) != "sigmoid":
        raise ValueError("Resume checkpoint classifier output must use sigmoid.")

    backbone.trainable = False
    for layer in backbone.layers:
        layer.trainable = False
    return model


def _load_resume_model(
    model_path: Path,
    *,
    image_size: tuple[int, int],
    learning_rate: float,
) -> tf.keras.Model:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume because no checkpoint exists at {model_path}."
        )

    _validate_resume_metadata(model_path, image_size)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        raise ValueError(
            f"Could not load resume checkpoint at {model_path}: {exc}"
        ) from exc

    _validate_resume_model(model, image_size)
    _compile_model(model, learning_rate)
    return model


def _validation_loss(
    model: tf.keras.Model,
    validation_dataset: tf.data.Dataset,
) -> float:
    results = model.evaluate(validation_dataset, verbose=0, return_dict=True)
    try:
        loss = float(results["loss"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Could not determine resume checkpoint validation loss."
        ) from exc
    if not math.isfinite(loss):
        raise ValueError("Resume checkpoint validation loss must be finite.")
    return loss


def _enable_fine_tuning(
    model: tf.keras.Model,
    *,
    fine_tune_layers: int,
    learning_rate: float,
) -> None:
    if fine_tune_layers <= 0:
        raise ValueError("fine_tune_layers must be positive.")

    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - fine_tune_layers)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = (
            index >= cutoff
            and not isinstance(layer, tf.keras.layers.BatchNormalization)
        )
    _compile_model(model, learning_rate)


def _merge_histories(
    initial: tf.keras.callbacks.History,
    additional: tf.keras.callbacks.History,
) -> tf.keras.callbacks.History:
    for metric_name, values in additional.history.items():
        initial.history.setdefault(metric_name, []).extend(values)
    initial.epoch.extend(additional.epoch)
    return initial


def train_model(
    dataset_dir: str | Path = DATASET_DIR,
    model_path: str | Path = MODEL_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    image_size: tuple[int, int] = IMAGE_SIZE,
    seed: int = RANDOM_SEED,
    weights: str | None = "imagenet",
    fine_tune_epochs: int = 0,
    fine_tune_learning_rate: float = FINE_TUNE_LEARNING_RATE,
    verbose: int = 1,
    *,
    fine_tune_layers: int = 20,
    resume: bool = False,
    initial_epoch: int = 0,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train the classifier and save the best validation checkpoint.

    Fine-tuning is optional and disabled by default for CPU-friendly operation.
    When enabled, only the final backbone layers (excluding batch-normalization
    layers) are unfrozen after the frozen-head training phase.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if fine_tune_epochs < 0:
        raise ValueError("fine_tune_epochs cannot be negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if initial_epoch < 0:
        raise ValueError("initial_epoch cannot be negative.")
    if not resume and initial_epoch:
        raise ValueError("initial_epoch can only be used with resume=True.")
    if initial_epoch >= epochs:
        raise ValueError("initial_epoch must be less than epochs.")

    tf.keras.utils.set_random_seed(seed)
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    train_dataset, validation_dataset, _ = load_datasets(
        dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        class_names=CLASS_NAMES,
        seed=seed,
    )
    if resume:
        model = _load_resume_model(
            destination,
            image_size=image_size,
            learning_rate=learning_rate,
        )
    else:
        model = build_model(
            image_size=image_size,
            learning_rate=learning_rate,
            weights=weights,
        )

    # Write the sidecar before fitting so even an interrupted first checkpoint
    # retains its preprocessing and class-label contract.
    save_model_metadata(
        destination,
        class_names=CLASS_NAMES,
        image_size=image_size,
    )
    if resume:
        initial_best = _validation_loss(model, validation_dataset)
        if verbose:
            print(f"Resume checkpoint validation loss: {initial_best:.4f}")
    else:
        initial_best = None

    callbacks = _training_callbacks(
        destination,
        initial_best=initial_best,
        initial_epoch=initial_epoch,
    )
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    if fine_tune_epochs:
        _enable_fine_tuning(
            model,
            fine_tune_layers=fine_tune_layers,
            learning_rate=fine_tune_learning_rate,
        )
        fine_tune_best = _validation_loss(model, validation_dataset)
        fine_tune_callbacks = _training_callbacks(
            destination,
            initial_best=fine_tune_best,
            initial_epoch=epochs,
        )
        fine_tune_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            initial_epoch=epochs,
            epochs=epochs + fine_tune_epochs,
            callbacks=fine_tune_callbacks,
            verbose=verbose,
        )
        history = _merge_histories(history, fine_tune_history)

    if not destination.is_file():
        # ModelCheckpoint normally creates this. The explicit save is a safe
        # fallback for unusual callback behavior or mocked training in tests.
        model.save(destination)
    return model, history


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the EfficientNetB0 coral bleaching classifier.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --model-path instead of building a new model.",
    )
    parser.add_argument(
        "--initial-epoch",
        type=int,
        default=0,
        help="Number of completed epochs when resuming.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use 'none' only for offline/debug runs; ImageNet is the default.",
    )
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--fine-tune-layers", type=int, default=20)
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=FINE_TUNE_LEARNING_RATE,
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    weights = None if args.weights == "none" else args.weights
    train_model(
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        weights=weights,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        fine_tune_layers=args.fine_tune_layers,
        resume=args.resume,
        initial_epoch=args.initial_epoch,
    )
    print(f"Best model saved to: {args.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
