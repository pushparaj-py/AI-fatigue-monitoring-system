"""
train_yawn_model.py

Trains a CNN to classify mouth/face images as Yawning or NotYawning using
frames extracted from the YawDD Mirror dataset (datasets/YawDD_frames/).

Includes:
- Class weighting to handle the Yawning/NotYawning class imbalance
  (~13,487 Yawning vs ~31,356 NotYawning frames)
- Epoch checkpointing so an interrupted run can be resumed, same as
  the eye-state model training script.

Outputs:
- Trained model saved to models/yawn_model/
- Printed accuracy, precision, recall, F1-score, and confusion matrix
  (all computed from real evaluation on a held-out test set)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join("datasets", "YawDD_frames")
IMG_SIZE = 64  # consistent with the eye model, for a lightweight CPU-friendly model
BATCH_SIZE = 32
EPOCHS = 15
MODEL_SAVE_DIR = os.path.join("models", "yawn_model")
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "checkpoint.keras")
CLASS_NAMES = ["NotYawning", "Yawning"]  # label 0 = NotYawning, label 1 = Yawning


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
                f"Check that datasets/YawDD_frames/ contains 'Yawning' and 'NotYawning' subfolders."
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

    images = np.array(images, dtype="float32") / 255.0
    images = np.expand_dims(images, axis=-1)
    labels = np.array(labels, dtype="int32")

    return images, labels


print("=" * 60)
print("RASD Yawn-State CNN Training")
print("=" * 60)
print("Loading dataset...")
X, y = load_dataset(DATASET_DIR, IMG_SIZE)
print(f"Total images loaded: {len(X)}")
print(f"Class distribution — NotYawning: {np.sum(y == 0)}, Yawning: {np.sum(y == 1)}")

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
# Step 3: Compute class weights to handle imbalance
# ---------------------------------------------------------------------------
class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train,
)
class_weight_dict = {0: class_weight_values[0], 1: class_weight_values[1]}
print(f"Computed class weights (to offset imbalance): {class_weight_dict}")

# ---------------------------------------------------------------------------
# Step 4: Build the CNN (or resume from a saved checkpoint, if one exists)
# ---------------------------------------------------------------------------
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

if os.path.exists(CHECKPOINT_PATH):
    print(f"\nFound existing checkpoint at {CHECKPOINT_PATH} — resuming from it "
          f"instead of starting fresh.\n")
    model = tf.keras.models.load_model(CHECKPOINT_PATH)
else:
    print("\nNo checkpoint found — building a new model from scratch.\n")
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
        layers.Dense(1, activation="sigmoid"),  # binary output: 0=NotYawning, 1=Yawning
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

model.summary()

checkpoint_callback = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    save_freq="epoch",
    verbose=1,
)

# ---------------------------------------------------------------------------
# Step 5: Train (with class weighting applied)
# ---------------------------------------------------------------------------
print("\nStarting training...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_callback],
)

# ---------------------------------------------------------------------------
# Step 6: Evaluate on the held-out test set (real metrics only)
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
print("            Predicted NotYawning  Predicted Yawning")
print(f"Actual NotYawning   {cm[0][0]:>6}              {cm[0][1]:>6}")
print(f"Actual Yawning      {cm[1][0]:>6}              {cm[1][1]:>6}")

# ---------------------------------------------------------------------------
# Step 7: Save the final trained model (separate from the epoch checkpoint)
# ---------------------------------------------------------------------------
final_model_path = os.path.join(MODEL_SAVE_DIR, "yawn_model.keras")
model.save(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")
print(f"(Epoch checkpoint remains at: {CHECKPOINT_PATH})")