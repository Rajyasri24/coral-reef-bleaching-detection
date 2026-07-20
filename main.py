"""Command-line entry point for the coral reef bleaching project."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from src.config import (
    BATCH_SIZE,
    CONFUSION_MATRIX_PATH,
    DATASET_DIR,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    PROJECT_ROOT,
)


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for an argparse option."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer for an argparse option."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point value."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def probability(value: str) -> float:
    """Parse a probability strictly between zero and one."""
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and less than 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or run the coral bleaching classifier."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train", help="Train the EfficientNetB0 classifier."
    )
    train_parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DIR,
        help=f"Dataset root (default: {DATASET_DIR}).",
    )
    train_parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"Model output path (default: {MODEL_PATH}).",
    )
    train_parser.add_argument(
        "--epochs", type=positive_int, default=EPOCHS, help="Initial training epochs."
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --model instead of building a new model.",
    )
    train_parser.add_argument(
        "--initial-epoch",
        type=non_negative_int,
        default=0,
        help="Number of completed epochs when resuming (default: 0).",
    )
    train_parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=BATCH_SIZE,
        help="Images per training batch.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=positive_float,
        default=LEARNING_RATE,
        help="Initial optimizer learning rate.",
    )
    train_parser.add_argument(
        "--fine-tune-epochs",
        type=non_negative_int,
        default=0,
        help="Additional fine-tuning epochs (default: 0).",
    )
    train_parser.add_argument(
        "--fine-tune-learning-rate",
        type=positive_float,
        default=1e-5,
        help="Learning rate used during fine-tuning.",
    )
    train_parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Backbone weights; use 'none' for random initialization.",
    )

    evaluate_parser = commands.add_parser(
        "evaluate", help="Evaluate a trained model on the test split."
    )
    evaluate_parser.add_argument(
        "--dataset", type=Path, default=DATASET_DIR, help="Dataset root."
    )
    evaluate_parser.add_argument(
        "--model", type=Path, default=MODEL_PATH, help="Trained model path."
    )
    evaluate_parser.add_argument(
        "--batch-size", type=positive_int, default=BATCH_SIZE
    )
    evaluate_parser.add_argument(
        "--threshold",
        type=probability,
        default=0.5,
        help="Healthy-class decision threshold.",
    )
    evaluate_parser.add_argument(
        "--confusion-matrix",
        type=Path,
        default=CONFUSION_MATRIX_PATH,
        help="Path at which to save the confusion-matrix image.",
    )
    evaluate_parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Open the confusion-matrix plot after evaluation.",
    )

    predict_parser = commands.add_parser(
        "predict", help="Classify one coral reef image."
    )
    predict_parser.add_argument("image", type=Path, help="Image to classify.")
    predict_parser.add_argument(
        "--model", type=Path, default=MODEL_PATH, help="Trained model path."
    )

    commands.add_parser("app", help="Launch the Streamlit web application.")
    return parser


def run_train(args: argparse.Namespace) -> int:
    """Run model training using CLI arguments."""
    from src.train import train_model

    weights = None if args.weights == "none" else args.weights
    train_model(
        dataset_dir=args.dataset,
        model_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weights=weights,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        resume=args.resume,
        initial_epoch=args.initial_epoch,
    )
    print(f"Training complete. Best model: {args.model}")
    return 0


def print_evaluation(results: dict) -> None:
    """Print evaluation metrics and the confusion matrix clearly."""
    print("Evaluation results")
    for metric in ("loss", "accuracy", "precision", "recall", "f1"):
        if metric in results:
            print(f"  {metric.capitalize():<10} {float(results[metric]):.4f}")

    matrix = results.get("confusion_matrix")
    if matrix is None:
        return

    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    labels = list(results.get("class_names", ()))

    print("  Confusion matrix")
    if labels:
        print("    actual \\ predicted: " + "  ".join(map(str, labels)))
    for index, row in enumerate(matrix):
        prefix = f"    {labels[index]}: " if index < len(labels) else "    "
        print(prefix + "  ".join(str(value) for value in row))


def run_evaluate(args: argparse.Namespace) -> int:
    """Evaluate the configured model and print its metrics."""
    if not args.model.is_file():
        raise FileNotFoundError(
            f"No trained model found at {args.model}. Run 'python main.py train' first."
        )

    from src.evaluate import evaluate_model

    results = evaluate_model(
        model_path=args.model,
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        threshold=args.threshold,
        confusion_matrix_path=args.confusion_matrix,
        show_plot=args.show_plot,
    )
    print_evaluation(results)
    return 0


def run_predict(args: argparse.Namespace) -> int:
    """Classify one image and print its label and confidence."""
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model.is_file():
        raise FileNotFoundError(
            f"No trained model found at {args.model}. Run 'python main.py train' first."
        )

    from src.predict import predict_image

    label, confidence = predict_image(args.image, model_path=args.model)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    return 0


def run_app() -> int:
    """Launch Streamlit with the current Python interpreter."""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app.py"),
    ]
    return subprocess.run(command, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the selected project command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            return run_train(args)
        if args.command == "evaluate":
            return run_evaluate(args)
        if args.command == "predict":
            return run_predict(args)
        return run_app()
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
