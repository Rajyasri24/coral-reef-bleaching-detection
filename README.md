# Coral Reef Bleaching Detection

A lightweight, CPU-friendly deep learning project that classifies real underwater
coral photographs as **Healthy** or **Bleached**. It uses an ImageNet-pretrained
EfficientNetB0 model for transfer learning and provides both command-line tools and
a minimal Streamlit interface with prediction confidence.

The hosted endpoint is
[coral-reef-bleaching-detection-efficientnetb0.streamlit.app](https://coral-reef-bleaching-detection-efficientnetb0.streamlit.app/).

## Pipeline

```text
Labeled images -> resize and augment -> EfficientNetB0 transfer learning
               -> models/best_model.keras -> evaluation and prediction
               -> Streamlit upload interface
```

The evaluation command reports accuracy, precision, recall, F1-score, and a
confusion matrix.

## Final model results

The frozen-backbone EfficientNetB0 run completed 10 epochs. The best validation
checkpoint was produced by epoch 10.

| Metric | Validation | Test |
| --- | ---: | ---: |
| Loss | 0.1811 | 0.1433 |
| Accuracy | 92.22% | 94.16% |
| Macro precision | - | 94.13% |
| Macro recall | - | 94.21% |
| Macro F1-score | - | 94.15% |

The 257-image test split contains 135 Bleached and 122 Healthy images. The final
confusion matrix was `[[126, 9], [6, 116]]`, with rows representing actual
Bleached/Healthy labels and columns representing predictions.

![Final test-set confusion matrix](models/confusion_matrix.png)

These measurements describe the supplied dataset split. The data contains
offline augmentations and adjacent video frames, so results should not be treated
as an independent field-performance estimate until the data is regrouped and
split by original reef/site/video source.

## Project layout

```text
.
|-- dataset/
|   |-- train/{Healthy,Bleached}/
|   |-- valid/{Healthy,Bleached}/
|   `-- test/{Healthy,Bleached}/
|-- models/
|   |-- best_model.keras          # created by training
|   |-- best_model.metadata.json  # label/preprocessing contract
|   |-- evaluation_metrics.json   # created by evaluation
|   `-- confusion_matrix.png      # created by evaluation
|-- screenshots/                  # add approved UI screenshots manually
|-- src/
|   |-- config.py
|   |-- dataset_loader.py
|   |-- train.py
|   |-- evaluate.py
|   `-- predict.py
|-- app.py
|-- main.py
|-- .dockerignore
|-- .gitignore
|-- Dockerfile
|-- PROJECT_REPORT.md
|-- README.md
`-- requirements.txt
```

## Prerequisites

- A 64-bit installation of Python **3.11.x**
- `pip` and Python's `venv` module
- Enough free disk space for TensorFlow, the dataset, and the trained model
- Docker Desktop or Docker Engine only if using the container workflow

A GPU is not required. Training EfficientNetB0 on a CPU can take a while,
depending on the dataset size and processor. The first training run also needs an
internet connection if the ImageNet weights are not already cached.

## 1. Add the dataset manually

Download the [Coral Reefs Images dataset from Kaggle](https://www.kaggle.com/datasets/asfarhossainsitab/coral-reefs-images),
extract it, and arrange it inside this project as follows:

```text
dataset/
|-- train/
|   |-- Healthy/
|   `-- Bleached/
|-- valid/
|   |-- Healthy/
|   `-- Bleached/
`-- test/
    |-- Healthy/
    `-- Bleached/
```

Folder names are case-sensitive in Linux and Docker. The application never
downloads the dataset programmatically. The `dataset/` directory is excluded from
Git and Docker build contexts so the original images stay local.

## 2. Create the local environment

From the project root, create the required virtual environment named `venv`.

Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm the intended interpreter and TensorFlow installation:

```bash
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
```

The expected major/minor Python version is `3.11`, and the pinned TensorFlow
version is `2.16.2`.

## 3. Train and evaluate

The simplest workflow uses the top-level command dispatcher:

```bash
python main.py train
python main.py evaluate
```

Training reads `dataset/train` and `dataset/valid`, then saves the best validation
checkpoint to `models/best_model.keras`. Evaluation loads that checkpoint and
uses `dataset/test` only.

To continue an interrupted run after one completed epoch, use:

```bash
python main.py train --resume --initial-epoch 1 --epochs 10
```

With resume mode, `--initial-epoch` is the number of epochs already completed and
`--epochs` is the final epoch target, not the number of additional epochs. The
example therefore runs epochs 2 through 10. Resume mode requires the model file
selected by `--model` to exist. It verifies the EfficientNetB0 architecture,
single-output shape, input image size, and (when present) metadata class order.
The loaded backbone remains frozen and the model is compiled with a fresh Adam
optimizer at `--learning-rate`; optimizer and callback history are not restored.


For explicit training controls, run the module directly:

```bash
python -m src.train --epochs 10 --batch-size 32 --learning-rate 0.001
```

Useful optional arguments include `--dataset-dir`, `--model-path`, `--weights
imagenet|none`, `--resume`, `--initial-epoch`, and `--fine-tune-epochs`. Display
all supported options with:

```bash
python -m src.train --help
python -m src.evaluate --help
```

To save the test-set confusion matrix to a chosen location:

```bash
python -m src.evaluate --confusion-matrix confusion_matrix.png
```

## 4. Predict one image

After training, classify an individual coral photograph with either entry point:

```bash
python main.py predict path/to/coral.jpg
python -m src.predict path/to/coral.jpg
```

The result includes the predicted class and confidence score. Use
`python -m src.predict --help` to see options such as a custom model path.
The displayed score is derived from the model's sigmoid output and has not been
statistically calibrated; it should not be interpreted as a guaranteed
real-world probability of correctness.

## 5. Run the Streamlit app

Make sure `models/best_model.keras` exists, then start the interface:

```bash
python main.py app
```

Equivalently:

```bash
streamlit run app.py
```


## Git workflow

The ignore rules keep the virtual environment, dataset images, caches, and
experimental model binaries out of normal commits. The canonical inference
checkpoint and its companion artifacts are explicit exceptions and must be
committed for deployment.

```bash
git init
git add .
git commit -m "Initial project setup"
git branch -M main
git remote add origin <github-repository-url>
git push -u origin main
```

Before committing, inspect `git status --short` and ensure browser profiles,
virtual environments, dataset images, and credentials are absent. Also ensure
`models/best_model.keras`, `models/best_model.metadata.json`,
`models/evaluation_metrics.json`, and `models/confusion_matrix.png` are present.

The `main` branch is hosted at
[Rajyasri24/coral-reef-bleaching-detection](https://github.com/Rajyasri24/coral-reef-bleaching-detection)
and is connected to the Streamlit Community Cloud deployment. T
## Documentation and external links

| Resource | Link |
| Resource | Link |
| --- | --- |
| Original dataset source | [Kaggle: Coral Reefs Images](https://www.kaggle.com/datasets/asfarhossainsitab/coral-reefs-images) |
| Dataset Google Drive link | [https://drive.google.com/drive/folders/15cmbW6S-9nSbSJm6WrtOcKcExYmOMq_E?usp=drive_link] |
| Source-code repository | [GitHub: Rajyasri24/coral-reef-bleaching-detection](https://github.com/Rajyasri24/coral-reef-bleaching-detection) |
| Source-code Google Drive link | [https://drive.google.com/drive/folders/1THplwC2I6hrckfhmXMHp2SEiCydbvSzA?usp=drive_link]. |
| Streamlit application | [Hosted application](https://coral-reef-bleaching-detection-efficientnetb0.streamlit.app/) 
## Responsible use

This model is an educational screening tool, not a substitute for marine-biologist
assessment. Predictions can be affected by lighting, water clarity, camera color
balance, coral species, and images that differ from the training dataset.
