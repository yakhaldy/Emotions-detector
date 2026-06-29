# Emotions Detector 🎭

Real-time facial emotion recognition from a webcam video stream, built with a Convolutional Neural Network (CNN) trained from scratch in Keras/TensorFlow, with a MobileNetV2 transfer-learning variant as an optional extension.

## Overview

This project detects and classifies **7 facial emotions** — Happy, Sad, Angry, Surprise, Fear, Disgust, and Neutral — from a live video stream. It was built in two stages:

1. **Emotion Classification** — Train a CNN on a labeled facial-expression dataset (48x48 grayscale images).
2. **Live Face Tracking & Prediction** — Use OpenCV's Haar cascade to detect faces in a webcam feed, preprocess them, and feed them into the trained CNN to predict emotions in real time.

## Demo

```
$ python ./scripts/predict_live_stream.py

11:11:11s : Happy , 73%
11:11:12s : Happy , 93%
11:11:13s : Surprise , 71%
```

A green bounding box is drawn around the detected face in the live feed, with the predicted label and confidence overlaid on screen.

## Project Structure

```
project
├── data
│   ├── test.csv
│   ├── test_with_emotions.csv
│   ├── train.csv
│   └── xxx.csv
├── requirements.txt
├── README.md
├── results
│   ├── model
│   │   ├── learning_curves.png
│   │   ├── tensorboard.png
│   │   ├── final_emotion_model_arch.txt
│   │   └── final_emotion_model.keras
│   └── preprocessing_test
│       ├── image0.png
│       ├── image1.png
│       ├── ...
│       ├── image6.png
│       └── input_video.mp4
└── scripts
    ├── validation_loss_accuracy.py
    ├── predict_live_stream.py
    ├── predict.py
    ├── preprocess.py
    ├── train.py
    └── train_pretrained.py
```

## Model Architecture

A custom CNN built with **Keras** (TensorFlow backend), trained directly on 48x48 grayscale images.

```
Input: (48, 48, 1)

Block 1
  Conv2D(32, 3x3, padding="same", activation="relu")
  BatchNormalization()
  Conv2D(32, 3x3, padding="same", activation="relu")
  BatchNormalization()
  MaxPooling2D(2,2)        # 48 -> 24
  Dropout(0.25)

Block 2
  Conv2D(64, 3x3, padding="same", activation="relu")
  BatchNormalization()
  Conv2D(64, 3x3, padding="same", activation="relu")
  BatchNormalization()
  MaxPooling2D(2,2)        # 24 -> 12
  Dropout(0.25)

Block 3
  Conv2D(128, 3x3, padding="same", activation="relu")
  BatchNormalization()
  Conv2D(128, 3x3, padding="same", activation="relu")
  BatchNormalization()
  MaxPooling2D(2,2)        # 12 -> 6
  Dropout(0.25)

Classification head
  Flatten()
  Dense(512, activation="relu") -> BatchNorm -> Dropout(0.5)
  Dense(256, activation="relu") -> BatchNorm -> Dropout(0.3)
  Dense(7, activation="softmax")
```

**Total parameters:** 8,347,831 (2,781,799 trainable / 2,432 non-trainable from BatchNorm)

### Why this architecture
- **Double Conv per block** to extract richer features before downsampling.
- **BatchNormalization** after every Conv layer for more stable, faster training.
- **Dropout** (0.25 in conv blocks, 0.5/0.3 in the dense head) to fight overfitting on a relatively small dataset.
- **Data augmentation** (horizontal flip, ±10° rotation, 10% zoom) to improve generalization.
- Trained on the full dataset (no subsampling), targeting a CPU-friendly model that clears the 60%+ accuracy bar.

### Training setup
| Setting | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | sparse_categorical_crossentropy |
| Batch size | 32 |
| Max epochs | 60 |
| Early stopping | `monitor="val_loss"`, patience=10, restore best weights |
| LR scheduling | `ReduceLROnPlateau`, factor=0.5, patience=4, min_lr=1e-6 |
| Monitoring | TensorBoard (per-epoch logs + histograms) |
| Augmentation | horizontal flip, rotation ±10°, zoom 10% |

### Final results
| Metric | Value |
|---|---|
| Final test loss | 0.9524 |
| **Final test accuracy** | **64.68%** |

Training was stopped automatically by early stopping once validation loss stopped improving, well before the model started overfitting — see `results/model/learning_curves.png` and `results/model/tensorboard.png`.

## Preprocessing Pipeline

**For training/test data (`preprocess.py`):**
- Reads pixel strings from the CSV (`load_data`).
- `preprocess_pixels()` parses each string into a `(48, 48, 1)` float32 array, normalized to `[0, 1]`.

**For live webcam frames (`predict_live_stream.py`):**
1. Capture frame from webcam (`cv2.VideoCapture(0)`), or fall back to `results/preprocessing_test/input_video.mp4` if no webcam is found.
2. Convert frame to grayscale.
3. Detect faces with OpenCV's Haar cascade (`haarcascade_frontalface_default.xml`), `scaleFactor=1.3`, `minNeighbors=5`.
4. Select the largest detected face and crop it.
5. Resize the crop to `48x48`.
6. Normalize to `[0, 1]` and reshape to `(1, 48, 48, 1)`.
7. Run inference through the trained CNN, capped at **1 prediction per second**.
8. Draw a bounding box and label/confidence overlay on the live frame.

Example preprocessing outputs are saved in `results/preprocessing_test/` (`image0.png` → `image6.png`).

## Optional: Transfer Learning (`train_pretrained.py`)

As an extension, a second model uses **MobileNetV2** pretrained on ImageNet:
- Grayscale images are converted to pseudo-RGB by repeating the single channel 3x (`48, 48, 3`).
- MobileNetV2 backbone (`include_top=False`, ImageNet weights).
- Custom head: `GlobalAveragePooling2D → Dense(256) → BatchNorm → Dropout(0.5) → Dense(128) → BatchNorm → Dropout(0.3) → Dense(7, softmax)`.
- **Phase 1:** backbone frozen, train head only (20 epochs).
- **Phase 2:** unfreeze last 30 layers, fine-tune at a low learning rate (1e-5, 30 epochs).

This compares a model trained from scratch (8.35M params, grayscale input) against a fine-tuned ImageNet backbone (RGB-adapted input) on the same task.

## Installation

```bash
git clone https://github.com/yakhaldy/Emotions-detector
cd emotions-detector
pip install -r requirements.txt
```

**Dependencies:**
```
numpy>=1.24
matplotlib>=3.8
scikit-learn>=1.3
opencv-python>=4.8
pandas>=2.0
tensorflow==2.20.0
keras==3.12.2
```

## Usage

**Train the model from scratch:**
```bash
python ./scripts/train.py
```

**Train the transfer-learning variant (optional):**
```bash
python ./scripts/train_pretrained.py
```

**Evaluate on the test set:**
```bash
python ./scripts/predict.py
# Accuracy on test set: 64%
```

**Run live webcam emotion detection:**
```bash
python ./scripts/predict_live_stream.py
```

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir results/logs
```

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV (Haar cascade face detection)
- TensorBoard
- MobileNetV2 (transfer learning, optional)

## Key Learnings
- Designing and tuning a CNN from scratch for facial expression classification.
- Using BatchNorm, Dropout, and data augmentation together to control overfitting.
- Using early stopping and learning-rate scheduling to stop training at the right point.
- Real-time face detection and preprocessing with OpenCV Haar cascades.
- Comparing a from-scratch CNN against a fine-tuned pretrained backbone (MobileNetV2).
- Monitoring and interpreting training behavior with TensorBoard and learning curves.

## Authors
`yakhaldy`
`saljaoui`
