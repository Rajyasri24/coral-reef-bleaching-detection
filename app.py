"""Streamlit interface for coral reef health classification."""

import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import (
    CONFUSION_MATRIX_PATH,
    EVALUATION_METRICS_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
)


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


st.set_page_config(
    page_title="Coral Reef Health Assessment",
    page_icon="🪸",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_model(model_path: str):
    """Load and cache the trained Keras model for Streamlit reruns."""
    from src.predict import load_trained_model

    return load_trained_model(Path(model_path))


def display_model_path(model_path: Path) -> str:
    """Return a concise model path for messages shown in the UI."""
    try:
        return str(model_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(model_path)


def read_evaluation_metrics(metrics_path: Path) -> dict | None:
    """Read saved test metrics without importing the training stack."""
    if not metrics_path.is_file():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return metrics if isinstance(metrics, dict) else None


def show_model_performance(metrics: dict) -> None:
    """Show compact, clearly scoped test-set performance details."""
    with st.expander("Validated model performance"):
        columns = st.columns(4)
        values = (
            ("Accuracy", metrics.get("accuracy")),
            ("Macro precision", metrics.get("precision")),
            ("Macro recall", metrics.get("recall")),
            ("Macro F1", metrics.get("f1", metrics.get("f1_score"))),
        )
        for column, (label, value) in zip(columns, values):
            shown_value = f"{float(value):.1%}" if value is not None else "N/A"
            column.metric(label, shown_value)

        samples = metrics.get("samples")
        if samples is not None:
            st.caption(f"Measured on the supplied {int(samples)}-image test split.")
        confusion_matrix_path = Path(CONFUSION_MATRIX_PATH)
        if confusion_matrix_path.is_file():
            st.image(
                str(confusion_matrix_path),
                caption="Test-set confusion matrix",
                use_column_width=True,
            )
        st.caption(
            "These dataset-level metrics are separate from the confidence "
            "reported for an uploaded image."
        )


def read_uploaded_image(uploaded_file) -> Image.Image:
    """Decode an uploaded image, correct its orientation, and convert to RGB."""
    upload_size = getattr(uploaded_file, "size", None)
    if upload_size is not None and upload_size > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded image must be 10 MB or smaller.")

    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("The uploaded image has too many pixels to process safely.")
    image.load()
    return ImageOps.exif_transpose(image).convert("RGB")


def show_prediction(label: str, confidence: float) -> None:
    """Render a prediction with a label-specific status message."""
    normalized_label = label.strip().lower()
    message = f"Prediction: **{label}**"

    if normalized_label == "healthy":
        st.success(message)
    elif normalized_label == "bleached":
        st.warning(message)
    else:
        st.info(message)

    st.metric("Confidence", f"{confidence:.1%}")
    st.progress(min(max(float(confidence), 0.0), 1.0))


def main() -> None:
    """Render the coral bleaching detection application."""
    model_path = Path(MODEL_PATH)
    model_ready = model_path.is_file()
    shown_model_path = display_model_path(model_path)
    evaluation_metrics = read_evaluation_metrics(Path(EVALUATION_METRICS_PATH))

    st.title("🪸 Coral Reef Health Assessment")
    st.write(
        "Upload an underwater coral image to classify it as "
        "**Healthy** or **Bleached**."
    )

    with st.sidebar:
        st.header("About")
        st.caption(
            "This classifier uses an EfficientNetB0 model trained on labeled "
            "coral reef photographs."
        )
        if model_ready:
            st.success("Model ready")
        else:
            st.warning("Model not trained")
        if evaluation_metrics is not None:
            st.divider()
            st.metric(
                "Test accuracy",
                f"{float(evaluation_metrics['accuracy']):.1%}",
            )
            st.caption(
                f"Validated on {int(evaluation_metrics.get('samples', 0))} images"
            )

    if not model_ready:
        st.warning(
            f"No trained model was found at `{shown_model_path}`. "
            "Run `python main.py train` first, then refresh this page."
        )

    if evaluation_metrics is not None:
        show_model_performance(evaluation_metrics)

    uploaded_file = st.file_uploader(
        "Choose a coral reef image",
        type=("jpg", "jpeg", "png"),
        help="Supported formats: JPG and PNG.",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.", icon="ℹ️")
        return

    try:
        image = read_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("The uploaded file could not be read as a valid image.")
        return

    st.image(image, caption=uploaded_file.name, use_column_width=True)

    if st.button(
        "Analyze image",
        type="primary",
        use_container_width=True,
        disabled=not model_ready,
    ):
        try:
            with st.spinner("Analyzing coral image..."):
                from src.predict import predict_image

                model = get_model(str(model_path))
                label, confidence = predict_image(image, model=model)
        except (FileNotFoundError, OSError) as exc:
            st.error(f"The trained model could not be loaded: {exc}")
        except Exception as exc:  # Streamlit should report inference failures cleanly.
            st.error(f"Prediction failed: {exc}")
        else:
            show_prediction(label, confidence)
            st.caption(
                "This prediction is a screening aid and should be interpreted "
                "alongside expert ecological assessment."
            )


if __name__ == "__main__":
    main()
