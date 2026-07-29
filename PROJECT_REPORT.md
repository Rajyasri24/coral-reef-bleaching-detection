# Coral Reef Bleaching Detection and Health Assessment Using Deep Learning

# I. Introduction

## 1.1 Background

Coral bleaching is the visible loss of color that occurs when corals are stressed and lose or expel their symbiotic algae. Image-based screening can help researchers review large collections of underwater photographs more quickly, although an automated classifier cannot replace ecological field measurements or expert assessment.

This project implements a lightweight, end-to-end deep learning system for coral-health screening. It accepts a real underwater image, applies the same image preparation used during training, and predicts one of two labels: **Bleached** or **Healthy**. An ImageNet-pretrained EfficientNetB0 network supplies learned visual features, while a small binary classification head is trained on the coral dataset. The selected model is exposed through both command-line inference and a Streamlit upload interface.

## 1.2 Problem statement

Manual review of underwater coral imagery can be time-consuming and subject to observer variation. The engineering problem is to build a reproducible, CPU-compatible image-classification pipeline that:

1. validates and loads a labeled coral dataset;
2. trains a binary deep learning model without requiring a GPU;
3. measures performance on a held-out test directory;
4. preserves the model's preprocessing and label semantics for reliable inference; and
5. provides a simple user interface suitable for later free deployment.

## 1.3 Objectives

The project objectives were to:

- classify 224 x 224 RGB coral images as Bleached or Healthy;
- use EfficientNetB0 transfer learning to keep training practical on a local CPU;
- use online augmentation to improve robustness to modest image transformations;
- save the checkpoint with the lowest observed validation loss;
- report accuracy, macro precision, macro recall, macro F1-score, and a confusion matrix;
- implement reusable single-image inference with a displayed confidence score;
- create a clean Streamlit frontend; and
- prepare Docker and documentation artifacts for a later deployment stage.

## 1.4 Scope

The delivered system is an **image-level binary classifier**. It does not localize coral, segment bleached regions, estimate bleaching severity, identify coral species, or infer the ecological cause of bleaching. Its output should be treated as an educational screening result, not as a diagnosis or field-survey measurement.

---

# II. Methodology

## 2.1 End-to-end workflow

```text
Kaggle image dataset
        |
        v
Directory and image-integrity checks
        |
        v
RGB decoding -> bilinear resize to 224 x 224 -> float32 [0, 255]
        |
        +-----------------------------+
        | training split only         |
        v                             |
Online geometric/color augmentation  |
        |                             |
        +-----------------------------+
        v
Frozen ImageNet EfficientNetB0 -> global average pooling -> dropout -> sigmoid
        |
        v
Validation-loss checkpoint selection -> models/best_model.keras
        |
        +--------------------+
        |                    |
        v                    v
Held-out test evaluation     CLI / Streamlit inference
```

The code is separated by responsibility: configuration in `src/config.py`, input handling in `src/dataset_loader.py`, training in `src/train.py`, evaluation in `src/evaluate.py`, inference in `src/predict.py`, and the user interface in `app.py`. `main.py` provides a single command dispatcher.

## 2.2 Data collection

The images came from the externally curated [Coral Reefs Images dataset on Kaggle](https://www.kaggle.com/datasets/asfarhossainsitab/coral-reefs-images). The project did not collect the photographs itself and therefore does not claim control over the original camera settings, reef sites, species distribution, labeling protocol, or train/test provenance. The dataset was downloaded and placed locally rather than fetched programmatically.

The supplied directory organization was retained:

```text
dataset/
|-- train/
|   |-- Bleached/
|   `-- Healthy/
|-- valid/
|   |-- Bleached/
|   `-- Healthy/
`-- test/
    |-- Bleached/
    `-- Healthy/
```

The directory names define the supervised labels. The implementation fixes the class order as `("Bleached", "Healthy")` instead of relying implicitly on a changing folder order. Consequently, model output index 1 is interpreted as the probability of **Healthy**.

## 2.3 Data preprocessing

The TensorFlow pipeline uses `image_dataset_from_directory()` with inferred binary labels and performs the following operations:

- decode each image as three-channel RGB;
- resize to 224 x 224 pixels using bilinear interpolation;
- batch 16 images at a time;
- cast pixels to `float32`;
- retain the `[0, 255]` input value range expected by the Keras EfficientNet implementation; and
- prefetch batches using `tf.data.AUTOTUNE`.

EfficientNetB0 contains its own rescaling/normalization layer. The loader calls the documented EfficientNet preprocessing API, but it does **not** divide input pixels by 255 a second time. The inference module follows the same RGB conversion, bilinear resizing, data type, and value-range contract so that training and deployed predictions receive consistent input.

Training data are shuffled using random seed 42. Validation and test data are not shuffled, which keeps evaluation order deterministic. Model metadata stored beside the checkpoint records the architecture, class names, positive class, input size, color mode, value range, and normalization contract.

## 2.4 Exploratory data analysis (EDA)

EDA for the delivered pipeline focused on dataset structure, class counts, split proportions, and file integrity. The audit found 10,382 valid JPEG images; no corrupt JPEG was retained in the reported inventory.

| Split | Bleached | Healthy | Total | Share of all data | Bleached share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 4,980 | 4,682 | 9,662 | 93.06% | 51.54% |
| Validation | 240 | 223 | 463 | 4.46% | 51.84% |
| Test | 135 | 122 | 257 | 2.48% | 52.53% |
| **Total** | **5,355** | **5,027** | **10,382** | **100.00%** | **51.58%** |

The class balance is close enough that the experiment did not use class weights or resampling. However, the validation and test sets are small relative to the training set, especially the 257-image test split. The current EDA does not establish independence by reef, site, original video, or capture session. A perceptual-duplicate and source-group audit is therefore an important future requirement.

No separate EDA notebook, species analysis, lighting histogram, or geographic analysis is claimed. Those analyses require metadata that are not represented in the folder-based loader.

## 2.5 Feature engineering and data augmentation

No handcrafted texture, color, edge, or shape descriptors were engineered. Feature learning is performed by the convolutional EfficientNetB0 backbone initialized with ImageNet weights. Global average pooling converts the final spatial feature maps into a compact feature vector for binary classification.

During training only, the model applies these online augmentations:

- random horizontal flipping;
- random rotation with factor 0.05 (approximately up to 18 degrees in either direction);
- random zoom with factor 0.10; and
- random contrast adjustment with factor 0.10.

The augmentation layers are embedded in the model but are inactive during validation, testing, and prediction. The existing dataset is also reported to contain offline augmentations and adjacent frames from videos. That provenance creates a possible near-duplicate/source-leakage risk across splits; online augmentation does not solve that risk.

## 2.6 Model description

### 2.6.1 Model building and architecture

| Stage | Configuration | Purpose |
| --- | --- | --- |
| Input | 224 x 224 x 3 RGB | Fixed image tensor expected by training and inference |
| Augmentation | Flip, rotation, zoom, contrast | Training-time robustness |
| Backbone | EfficientNetB0, ImageNet weights, `include_top=False` | Transfer-learned visual feature extraction |
| Pooling | GlobalAveragePooling2D | Converts feature maps to one vector |
| Regularization | Dropout 0.20 | Reduces dependence on individual activations |
| Output | Dense(1), sigmoid | Estimates `P(Healthy)` |

The saved model contains 4,050,852 parameters. With the EfficientNetB0 backbone frozen, 1,281 classifier-head parameters are trainable and 4,049,571 parameters are non-trainable. The final reported run used the frozen-backbone configuration; the optional fine-tuning path in the code was not used for these results.

### 2.6.2 Training configuration

| Setting | Value |
| --- | --- |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Objective | Binary cross-entropy |
| Batch size | 16 |
| Epoch target | 10 |
| Random seed | 42 |
| Positive class | Healthy |
| Decision threshold | 0.50 |
| Selection metric | Minimum validation loss |

Training used three callbacks:

- `ModelCheckpoint` saved only an improvement in validation loss;
- early stopping used validation loss with patience 3 and restoration of the best weights; and
- `ReduceLROnPlateau` halved the learning rate after two non-improving epochs, down to a minimum of `1e-6`.

The first completed checkpoint was preserved after an interrupted epoch-1 run. Resume mode reloaded and validated that checkpoint, evaluated its validation loss, and used that loss as the baseline for subsequent checkpointing. The backbone remained frozen and the Adam optimizer was freshly compiled on resume; optimizer and callback history were not restored.

### 2.6.3 Model selection

The final checkpoint was selected by **validation loss**, not by test accuracy. Validation loss improved from 0.2259 at the first completed checkpoint to 0.1811 at epoch 10, an absolute decrease of 0.0448 (approximately 19.8% relative). The epoch-10 checkpoint was therefore retained as `models/best_model.keras`.

No alternative neural architecture, ablation study, hyperparameter sweep, or cross-validation experiment was run. Accordingly, this report does **not** claim that EfficientNetB0 outperformed MobileNet, ResNet, VGG, or any other architecture. The only valid comparisons available are a derived majority-class baseline and a descriptive comparison between two checkpoints of the same EfficientNetB0 run.

### 2.6.4 Evaluation procedure

The selected checkpoint was loaded without its training compiler state and evaluated once per image on the non-shuffled test split. A Healthy prediction is produced when `P(Healthy) >= 0.50`; otherwise the prediction is Bleached. The evaluator computes the confusion matrix directly and derives per-class precision, recall, and F1-score. Reported precision, recall, and F1 are **macro averages**, so Bleached and Healthy receive equal weight regardless of support.

For a class, precision measures the fraction of its predictions that are correct, recall measures the fraction of its true examples recovered, and F1 is their harmonic mean. Accuracy is the fraction of all correct predictions. The confusion-matrix row order is actual `Bleached, Healthy`, and the column order is predicted `Bleached, Healthy`.

## 2.7 Tools and libraries

| Tool / library | Version or role | Actual use in this project |
| --- | --- | --- |
| Python | 3.11 | Application language and local virtual environment |
| TensorFlow / Keras | 2.16.2 | Dataset pipeline, EfficientNetB0, training, checkpoint loading, and inference |
| NumPy | 1.26.4 | Array conversion and metric calculations |
| OpenCV (headless) | 4.10.0.84 | Robust image decoding and RGB color conversion in inference |
| Pillow | 10.4.0 | Streamlit upload decoding, EXIF orientation correction, and RGB conversion |
| Matplotlib | 3.9.2 | Confusion-matrix visualization |
| Streamlit | 1.37.1 | Browser-based frontend |
| Docker | Python 3.11 slim image | Reproducible inference container configuration |
| Git | Local version-control workflow | Prepared for a future hosted repository |

`pandas==2.2.2` and `scikit-learn==1.5.1` are pinned in `requirements.txt`, but the current executable pipeline does not import them: metric calculation is implemented with NumPy, and no DataFrame-based analysis is part of the delivered code.

---

# III. Results and Discussion

## 3.1 About the dataset

The experiment used 9,662 training images, 463 validation images, and 257 test images. Across all splits, Bleached images account for 51.58% and Healthy images account for 48.42%, so this is not a severely imbalanced dataset. The largest-class proportion on the test set is 52.53%, which establishes the accuracy of an always-Bleached majority baseline.

Although every inventoried JPEG passed the file-validity check, file validity is different from scientific independence. Project notes indicate that the collection contains offline augmentations and adjacent video frames. Images derived from the same source can be visually very similar even when they are separate valid files. If related images appear in different splits, the measured score can overestimate performance on a genuinely new reef, camera, site, or survey.

## 3.2 Final training and validation results

| Epoch-10 metric | Value |
| --- | ---: |
| Training accuracy | 94.44% |
| Training loss | 0.1499 |
| Validation accuracy | 92.22% |
| Validation loss | 0.1811 |

The 2.22-percentage-point train/validation accuracy gap and the higher validation loss are consistent with a modest generalization gap, but a single run is insufficient to diagnose overfitting conclusively. The online augmentation and dropout are intended to reduce that gap.

## 3.3 Final test performance

| Test metric | Value |
| --- | ---: |
| Samples | 257 |
| Binary cross-entropy loss | 0.1433 |
| Accuracy | 0.9416 (94.16%) |
| Macro precision | 0.9413 (94.13%) |
| Macro recall | 0.9421 (94.21%) |
| Macro F1-score | 0.9415 (94.15%) |
| Correct / incorrect | 242 / 15 |

Per-class results derived from the final confusion matrix are:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Bleached | 95.45% | 93.33% | 94.38% | 135 |
| Healthy | 92.80% | 95.08% | 93.93% | 122 |

The final confusion matrix is:

```text
                         Predicted
                    Bleached  Healthy
Actual Bleached        126       9
Actual Healthy           6     116
```

Thus, 9 of 135 Bleached examples were classified as Healthy, and 6 of 122 Healthy examples were classified as Bleached. For monitoring use, the first error type may be especially important because a bleached sample could be missed. Threshold selection should eventually reflect the relative ecological cost of the two error types rather than automatically retaining 0.50.

## 3.4 Performance comparison

| System / checkpoint | Validation loss | Test loss | Test accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority baseline: always Bleached | Not applicable | Not applicable | 52.53% | 26.26% | 50.00% | 34.44% |
| First completed checkpoint (epoch 1) | 0.2259 | 0.1919 | 94.5525% | 94.5434% | 94.6570% | 94.5485% |
| Final selected checkpoint (epoch 10) | **0.1811** | **0.1433** | 94.16% | 94.13% | 94.21% | 94.15% |

The majority baseline is calculated from the test split, whose larger class is Bleached (135/257). The final model improves accuracy over that baseline by 41.63 percentage points and macro F1 by 59.71 percentage points, showing that it learned meaningful separation rather than simply predicting the more common class.

The first checkpoint's confusion matrix was `[[125, 10], [4, 118]]`. Its test accuracy was slightly higher than the final checkpoint's result: 243 rather than 242 correct images, a difference of one test example (0.39 percentage points). Conversely, binary cross-entropy on the test split decreased from 0.1918566 to 0.1433348, indicating better average probability assignment on that split. The final checkpoint also improved Bleached recall from 92.59% to 93.33% but reduced Healthy recall from 96.72% to 95.08%.

This checkpoint comparison must be interpreted carefully:

- the final checkpoint was selected because validation loss improved substantially, not because of the test result;
- the 257-image test set is too small for a one-image difference to establish superiority;
- inspecting the same test split at the first checkpoint and again at the final checkpoint makes the comparison descriptive, not independent model-selection evidence; and
- no alternate architecture was evaluated, so there is no empirical architecture comparison to report.

For a stronger experiment, the test split should remain untouched until all model and threshold choices are frozen, and uncertainty should be reported using a confidence interval or source-grouped bootstrap.

## 3.5 Screenshots and recorded outputs

### Final test confusion matrix

![Final test-set confusion matrix showing 126 correctly classified Bleached images, 116 correctly classified Healthy images, and 15 total errors](models/confusion_matrix.png)

*Figure 1. Final test-set confusion matrix generated by `src/evaluate.py`.*

### Streamlit upload workflow

The previously captured frontend image was intentionally removed pending privacy
review. Before final academic submission, add an approved screenshot showing the
model-ready interface and a representative prediction result. The missing image
is recorded as pending rather than being replaced with a fabricated example.

The local application is started with:

```powershell
python main.py app
```

It is then available at `http://localhost:8501` while that process remains active. This address is local to the machine and is not a permanent deployment URL.

## 3.6 Frontend, backend, and deployment readiness

The local inference pipeline is implemented end to end:

```text
uploaded JPEG/PNG
    -> size and pixel-count validation
    -> EXIF correction and RGB conversion
    -> 224 x 224 bilinear resize
    -> cached best_model.keras inference
    -> Healthy/Bleached label and displayed score
```

The frontend limits uploads to 10 MB and rejects images above 25 million decoded pixels. It displays a model-readiness indicator, uploaded image, label-specific result message, confidence metric, progress bar, and responsible-use notice. It also reads `models/evaluation_metrics.json` to show the 94.2% test accuracy in the sidebar and provides a collapsed performance section containing accuracy, macro precision, macro recall, macro F1, sample count, and the confusion matrix. These dataset-level metrics are explicitly separated from the score for an uploaded image. The model loader is cached across Streamlit reruns. There is no separate REST API because it was outside this project's minimal Streamlit scope; `src/predict.py` is the inference backend used by both the CLI and frontend.

The Dockerfile packages the inference application with Python 3.11, runs it as an unprivileged user, exposes port 8501, and includes a Streamlit health check. The code and required model artifact are hosted in GitHub, and a Streamlit Community Cloud endpoint has been created. External verification currently redirects anonymous users to Streamlit sign-in, so unrestricted public visibility is not yet established.

## 3.7 Challenges and limitations

1. **Possible source leakage:** Offline augmentations, adjacent video frames, or near-duplicates may cross directory splits. This can make held-out metrics optimistic. A split grouped by original reef, site, video, or survey is the highest-priority methodological improvement.
2. **Limited test size:** Only 257 images are in the test split. No confidence interval, repeated split, or cross-validation result is available.
3. **Repeated test inspection:** Both the first checkpoint and final checkpoint were evaluated on the same test directory. The final choice was correctly based on validation loss, but further iteration should use a new untouched test set or keep test results hidden until the end.
4. **No external validation:** The classifier has not been measured on photographs from a separate institution, reef, camera, or field campaign.
5. **Domain shift:** Lighting, depth, turbidity, white balance, coral species, viewpoint, and camera equipment can differ materially from the training distribution.
6. **Binary scope:** The model cannot express partial bleaching, severity, disease, dead coral, non-coral content, uncertainty classes, or spatial extent.
7. **Uncalibrated score:** The displayed number is the sigmoid probability assigned to the selected class (`p` for Healthy or `1-p` for Bleached). It has not been calibrated with temperature scaling, isotonic regression, a reliability diagram, expected calibration error, or Brier score; it should not be interpreted as a guaranteed real-world probability of correctness.
8. **Fixed threshold:** The 0.50 threshold was not optimized for a field-use cost function. Missing bleaching may justify a threshold selected for higher Bleached recall.
9. **No architecture comparison:** EfficientNetB0 was selected as a practical transfer-learning design, but the project did not experimentally compare it with other networks or a fine-tuned backbone.
10. **Limited explainability:** No Grad-CAM visualization or region-level evidence is currently shown to help a marine expert inspect why a decision was made.
11. **Dataset provenance and labels:** The project relies on external labels and does not independently verify the original annotation protocol, species coverage, geographic coverage, or licensing conditions beyond the supplied Kaggle source.
12. **Hosted access is currently restricted:** A permanent Streamlit endpoint exists, but anonymous requests redirect to sign-in. No anonymous prediction smoke test, hosted cold-start measurement, or public security/availability test is recorded.

## 3.8 Dataset, code, and Drive link status

| Resource | Link / status |
| --- | --- |
| Original dataset source | [Kaggle: Coral Reefs Images](https://www.kaggle.com/datasets/asfarhossainsitab/coral-reefs-images) |
| Dataset Google Drive link | [https://drive.google.com/drive/folders/15cmbW6S-9nSbSJm6WrtOcKcExYmOMq_E?usp=drive_link] |
| Source-code repository | [GitHub: Rajyasri24/coral-reef-bleaching-detection](https://github.com/Rajyasri24/coral-reef-bleaching-detection) |
| Source-code Google Drive link | [https://drive.google.com/drive/folders/1THplwC2I6hrckfhmXMHp2SEiCydbvSzA?usp=drive_link]. |
| Streamlit application | [Hosted application](https://coral-reef-bleaching-detection-efficientnetb0.streamlit.app/) 

---

# IV. Conclusion

## 4.1 Summary

This project delivered a compact coral-bleaching classification pipeline using a frozen, ImageNet-pretrained EfficientNetB0 backbone and a sigmoid classifier head. The implementation validates the expected dataset structure, performs consistent training and inference preprocessing, applies online augmentation, saves the best validation-loss checkpoint, evaluates the held-out test directory, supports command-line prediction, and provides a local Streamlit interface.

The final checkpoint achieved 94.16% test accuracy and 94.15% macro F1 on 257 supplied test images, with 242 correct predictions. This is substantially above the 52.53% majority-class accuracy baseline. The first checkpoint achieved a slightly higher test score by one image, while the final checkpoint had markedly better validation loss and was selected without using test accuracy as the criterion. These results demonstrate strong performance on the supplied split, but possible related-frame/preaugmentation leakage, the small test set, and the lack of external validation prevent treating 94.16% as a reliable field-performance estimate.

The core development, documentation, GitHub publication, and Streamlit deployment are complete. Remaining handoff items are enabling anonymous access if a public demonstration is required, recording a final hosted uploaded-image screenshot, and supplying the institution's requested dataset/code Drive links.

## 4.2 Future work

Recommended next steps, in priority order, are:

1. trace images to their original video/site/session and rebuild train, validation, and test splits by source group;
2. remove or contain offline-augmented and near-duplicate images within a single source group;
3. reserve a new untouched external test set and report confidence intervals;
4. compare EfficientNetB0 fairly with CPU-friendly alternatives such as MobileNetV3 and at least one residual architecture under the same grouped split;
5. evaluate controlled fine-tuning of the upper EfficientNet layers and tune hyperparameters using validation data only;
6. calibrate output probabilities and choose a decision threshold based on the cost of missed bleaching;
7. add Grad-CAM or a related explanation method and conduct expert review with marine scientists;
8. extend the label space to partial bleaching/severity or add segmentation when suitable annotations become available;
9. add automated tests and continuous integration for data contracts, model metadata, inference, and Streamlit startup;
10. upload the code and model artifact to an approved repository, deploy to Streamlit Community Cloud, smoke-test the public URL, and record a prediction-result screenshot; and
11. add the verified repository, Drive, and public application links to Section 3.8.

---

## Reproduction commands

From an activated Python 3.11 virtual environment in the project root:

```powershell
# Train the default frozen-backbone model
python main.py train

# Evaluate the selected checkpoint and regenerate the confusion matrix
python main.py evaluate

# Classify one local image
python main.py predict path\to\coral.jpg

# Launch the local Streamlit interface
python main.py app
```

The canonical artifacts are `models/best_model.keras`, `models/best_model.metadata.json`, `models/evaluation_metrics.json`, and `models/confusion_matrix.png`. An approved frontend screenshot should be added manually before final submission.
