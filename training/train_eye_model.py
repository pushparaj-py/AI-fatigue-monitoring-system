"""
train_eye_model.py

Trains a CNN to classify eye images as Open or Closed using the MRL Eye Dataset,
already organized into datasets/MRL/Open/ and datasets/MRL/Closed/ subfolders.

Outputs:
- Trained model saved to models/eye_model/
- Printed accuracy, precision, recall, F1-score, and confusion matrix
  (all computed from real evaluation on a held-out test set, not invented)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join("datasets", "MRL", "mrleyedataset")
IMG_SIZE = 64  # 64x64 is enough detail for eye open/closed classification,
               # and keeps training fast on CPU-only machines (per our
               # lightweight-hardware decision — no need for 224x224 here).
BATCH_SIZE = 32
EPOCHS = 15
MODEL_SAVE_DIR = os.path.join("models", "eye_model")
CLASS_NAMES = ["Close-Eyes", "Open-Eyes"]  # label 0 = Closed, label 1 = Open


# ---------------------------------------------------------------------------
# Step 1: Load and preprocess the dataset
# ---------------------------------------------------------------------------
def load_dataset(dataset_dir, img_size):
    images = []
    labels = []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected folder not found: {class_dir}. "
                f"Check that datasets/MRL/ contains 'Open' and 'Closed' subfolders."
            )

        filenames = os.listdir(class_dir)
        print(f"Loading {len(filenames)} images from '{class_name}'...")

        for fname in filenames:
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # skip unreadable/corrupt files
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(label_idx)

    images = np.array(images, dtype="float32") / 255.0  # normalize to 0-1
    images = np.expand_dims(images, axis=-1)  # add channel dimension (grayscale)
    labels = np.array(labels, dtype="int32")

    return images, labels


print("=" * 60)
print("RASD Eye-State CNN Training")
print("=" * 60)
print("Loading dataset...")
X, y = load_dataset(DATASET_DIR, IMG_SIZE)
print(f"Total images loaded: {len(X)}")

# ---------------------------------------------------------------------------
# Step 2: Split into train / validation / test (70 / 15 / 15)
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# ---------------------------------------------------------------------------
# Step 3: Build the CNN
# ---------------------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),  # binary output: 0=Closed, 1=Open
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------------
# Step 4: Train
# ---------------------------------------------------------------------------
print("\nStarting training...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# ---------------------------------------------------------------------------
# Step 5: Evaluate on the held-out test set (real metrics only)
# ---------------------------------------------------------------------------
print("\nEvaluating on test set...\n")
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
cm = confusion_matrix(y_test, y_pred)

print("=" * 60)
print("EVALUATION RESULTS (real, computed from test set)")
print("=" * 60)
print(report)
print("Confusion Matrix:")
print("            Predicted Closed  Predicted Open")
print(f"Actual Closed      {cm[0][0]:>6}           {cm[0][1]:>6}")
print(f"Actual Open        {cm[1][0]:>6}           {cm[1][1]:>6}")
# Note: CLASS_NAMES[0]="Close-Eyes" (label 0), CLASS_NAMES[1]="Open-Eyes" (label 1) —
# labels above reflect that ordering.

# ---------------------------------------------------------------------------
# Step 6: Save the trained model
# ---------------------------------------------------------------------------
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
model_path = os.path.join(MODEL_SAVE_DIR, "eye_model.keras")
model.save(model_path)
print(f"\nModel saved to: {model_path}")