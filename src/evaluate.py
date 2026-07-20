"""Evaluate a saved coral classifier and visualize its confusion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorflow as tf

try:
    from .config import (
        BATCH_SIZE,
        CLASS_NAMES,
        CONFUSION_MATRIX_PATH,
        DATASET_DIR,
        EVALUATION_METRICS_PATH,
        IMAGE_SIZE,
        MODEL_PATH,
        load_model_metadata,
    )
    from .dataset_loader import load_dataset, validate_dataset_structure
except ImportError:  # Support direct execution with ``python src/evaluate.py``.
    from config import (  # type: ignore
        BATCH_SIZE,
        CLASS_NAMES,
        CONFUSION_MATRIX_PATH,
        DATASET_DIR,
        EVALUATION_METRICS_PATH,
        IMAGE_SIZE,
        MODEL_PATH,
        load_model_metadata,
    )
    from dataset_loader import load_dataset, validate_dataset_structure  # type: ignore


def _class_names_for_model(model_path: Path) -> tuple[str, str]:
    metadata = load_model_metadata(model_path)
    if metadata is None:
        return CLASS_NAMES

    names = metadata.get("class_names")
    if not isinstance(names, list) or len(names) != 2:
        raise ValueError(
            f"Model metadata for {model_path} must contain two class_names."
        )
    stored_names = tuple(str(name) for name in names)
    if stored_names != CLASS_NAMES:
        raise ValueError(
            "Saved model class mapping does not match this project's dataset "
            f"mapping: {stored_names!r} != {CLASS_NAMES!r}."
        )
    return stored_names  # type: ignore[return-value]


def _collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    *,
    verbose: int,
) -> tuple[np.ndarray, np.ndarray]:
    label_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []

    total_batches = int(tf.data.experimental.cardinality(dataset).numpy())
    for batch_number, (images, labels) in enumerate(dataset, start=1):
        probabilities = np.asarray(model(images, training=False)).reshape(-1)
        true_labels = np.asarray(labels).reshape(-1).astype(np.int64)
        if probabilities.size != true_labels.size:
            raise ValueError(
                "Model output count does not match the test batch label count."
            )
        probability_batches.append(probabilities.astype(np.float64))
        label_batches.append(true_labels)
        if verbose > 1:
            suffix = f"/{total_batches}" if total_batches >= 0 else ""
            print(f"Evaluated batch {batch_number}{suffix}")

    if not label_batches:
        raise ValueError("The test dataset contains no images.")

    y_true = np.concatenate(label_batches)
    probabilities = np.concatenate(probability_batches)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model produced non-finite probabilities.")
    if np.any(probabilities < -1e-6) or np.any(probabilities > 1 + 1e-6):
        raise ValueError(
            "Model output is not a sigmoid probability in the range [0, 1]."
        )
    return y_true, np.clip(probabilities, 0.0, 1.0)


def calculate_metrics(
    y_true: np.ndarray,
    healthy_probabilities: np.ndarray,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate binary and macro classification metrics without sklearn."""

    names = tuple(str(name) for name in class_names)
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("Exactly two unique class names are required.")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1.")

    true_values = np.asarray(y_true).reshape(-1).astype(np.int64)
    probabilities = np.asarray(healthy_probabilities).reshape(-1).astype(float)
    if true_values.size == 0 or true_values.size != probabilities.size:
        raise ValueError("Labels and probabilities must be non-empty and equal length.")
    if not np.all(np.isin(true_values, (0, 1))):
        raise ValueError("All true labels must be either 0 or 1.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must all be finite.")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be in the range [0, 1].")

    predicted_values = (probabilities >= threshold).astype(np.int64)
    confusion_matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion_matrix, (true_values, predicted_values), 1)

    per_class: dict[str, dict[str, float | int]] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for class_index, class_name in enumerate(names):
        true_positive = int(confusion_matrix[class_index, class_index])
        false_positive = int(confusion_matrix[:, class_index].sum() - true_positive)
        false_negative = int(confusion_matrix[class_index, :].sum() - true_positive)
        support = int(confusion_matrix[class_index, :].sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1_score)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
        }

    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    loss = -np.mean(
        true_values * np.log(clipped)
        + (1 - true_values) * np.log(1 - clipped)
    )
    accuracy = float(np.trace(confusion_matrix) / true_values.size)
    macro_f1 = float(np.mean(f1_values))
    return {
        "loss": float(loss),
        "accuracy": accuracy,
        "precision": float(np.mean(precision_values)),
        "recall": float(np.mean(recall_values)),
        "f1_score": macro_f1,
        "f1": macro_f1,
        "threshold": float(threshold),
        "positive_class": names[1],
        "class_names": list(names),
        "confusion_matrix": confusion_matrix.tolist(),
        "per_class": per_class,
        "samples": int(true_values.size),
    }


def plot_confusion_matrix(
    confusion_matrix: Sequence[Sequence[int]],
    class_names: Sequence[str] = CLASS_NAMES,
    *,
    output_path: str | Path | None = CONFUSION_MATRIX_PATH,
    show: bool = False,
) -> None:
    """Save and optionally display a compact confusion-matrix figure."""

    import matplotlib.pyplot as plt

    matrix = np.asarray(confusion_matrix, dtype=np.int64)
    names = tuple(class_names)
    if matrix.shape != (len(names), len(names)):
        raise ValueError("Confusion matrix dimensions must match class_names.")

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Coral Health Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(names)),
        yticks=np.arange(len(names)),
        xticklabels=names,
        yticklabels=names,
    )

    midpoint = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > midpoint else "black",
            )
    figure.tight_layout()

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def save_evaluation_metrics(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Persist evaluation metrics as human- and machine-readable JSON."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def evaluate_model(
    model_path: str | Path = MODEL_PATH,
    dataset_dir: str | Path = DATASET_DIR,
    batch_size: int = BATCH_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
    threshold: float = 0.5,
    confusion_matrix_path: str | Path | None = CONFUSION_MATRIX_PATH,
    evaluation_metrics_path: str | Path | None = EVALUATION_METRICS_PATH,
    show_plot: bool = False,
    verbose: int = 1,
) -> dict[str, Any]:
    """Evaluate a saved model against the deterministic test split."""

    saved_model = Path(model_path)
    if not saved_model.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {saved_model}. Run training first."
        )
    class_names = _class_names_for_model(saved_model)
    validate_dataset_structure(dataset_dir, class_names=class_names)
    test_dataset = load_dataset(
        Path(dataset_dir) / "test",
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names,
        shuffle=False,
    )

    model = tf.keras.models.load_model(saved_model, compile=False)
    y_true, probabilities = _collect_predictions(
        model,
        test_dataset,
        verbose=verbose,
    )
    metrics = calculate_metrics(
        y_true,
        probabilities,
        class_names=class_names,
        threshold=threshold,
    )
    if confusion_matrix_path is not None or show_plot:
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            output_path=confusion_matrix_path,
            show=show_plot,
        )
    if evaluation_metrics_path is not None:
        save_evaluation_metrics(metrics, evaluation_metrics_path)
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained model on dataset/test.",
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--confusion-matrix",
        type=Path,
        default=CONFUSION_MATRIX_PATH,
        help="PNG output path (default: models/confusion_matrix.png).",
    )
    parser.add_argument(
        "--no-confusion-matrix",
        action="store_true",
        help="Do not save a confusion-matrix image.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=EVALUATION_METRICS_PATH,
        help="JSON output path (default: models/evaluation_metrics.json).",
    )
    parser.add_argument(
        "--no-metrics-json",
        action="store_true",
        help="Do not save evaluation metrics as JSON.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_path = None if args.no_confusion_matrix else args.confusion_matrix
    metrics_path = None if args.no_metrics_json else args.metrics_json
    metrics = evaluate_model(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        confusion_matrix_path=output_path,
        evaluation_metrics_path=metrics_path,
        show_plot=args.show,
        verbose=0 if args.quiet else 1,
    )
    print(json.dumps(metrics, indent=2))
    if output_path is not None:
        print(f"Confusion matrix saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
