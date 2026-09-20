"""
relabel_and_retrain_lstm.py

Diagnosis: the original LSTM training used video-SOURCE labels (a whole
video was "Yawning" or "Alert" based on its filename), but each video is
only a few seconds long and the person is only actually yawning for a
fraction of it. Since we sliced each video into many overlapping windows,
most "Yawning"-labeled windows didn't actually contain yawning behavior —
this is label noise, and it's why the LSTM was stuck near chance-level
accuracy (~51-63%).

Fix: re-derive each window's label from its OWN feature content instead
of the source video's filename. Specifically, a window is labeled
"Yawning" if its average yawn_cnn_score (feature index 5) exceeds a
threshold, meaning the frames actually WITHIN that window show yawning
behavior — not just that they came from a video that contains a yawn
somewhere.

This reuses the already-extracted features in training/sequence_data.npz,
so no slow video reprocessing is needed — just relabeling and retraining.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEQUENCE_DATA_PATH = os.path.join("training", "sequence_data.npz")
MODEL_SAVE_DIR = os.path.join("models", "lstm_model_v2")
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "checkpoint.keras")
BATCH_SIZE = 32
EPOCHS = 25
CLASS_NAMES = ["Alert", "Yawning"]

# Feature vector layout (from feature_extraction.py):
# [left_EAR, right_EAR, avg_EAR, MAR, eye_cnn_score, yawn_cnn_score]
YAWN_CNN_SCORE_INDEX = 5
YAWN_SCORE_THRESHOLD = 0.5  # average yawn_cnn_score above this -> label window as Yawning

# ---------------------------------------------------------------------------
# Step 1: Load existing sequence data and RELABEL based on actual content
# ---------------------------------------------------------------------------
print("=" * 60)
print("RASD LSTM Retraining with Content-Based Relabeling")
print("=" * 60)
print("Loading existing sequence data...")

data = np.load(SEQUENCE_DATA_PATH)
X, y_original = data["X"], data["y"]

print(f"Total sequences loaded: {len(X)}")
print(f"Original (video-source) label distribution — Alert: {np.sum(y_original == 0)}, "
      f"Yawning: {np.sum(y_original == 1)}")

# Recompute labels: average yawn_cnn_score across each window's frames
avg_yawn_scores = X[:, :, YAWN_CNN_SCORE_INDEX].mean(axis=1)
y = (avg_yawn_scores > YAWN_SCORE_THRESHOLD).astype("int32")

print(f"Content-based label distribution   — Alert: {np.sum(y == 0)}, Yawning: {np.sum(y == 1)}")
agreement = np.mean(y == y_original)
print(f"Agreement between original and content-based labels: {agreement:.2%}")
print("(A low agreement number here confirms the original video-source labels were noisy.)\n")

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
# Step 3: Compute class weights
# ---------------------------------------------------------------------------
class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train,
)
class_weight_dict = {0: class_weight_values[0], 1: class_weight_values[1]}
print(f"Computed class weights: {class_weight_dict}")

# ---------------------------------------------------------------------------
# Step 4: Build the LSTM (or resume from a checkpoint)
# ---------------------------------------------------------------------------
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
sequence_length = X.shape[1]
num_features = X.shape[2]

if os.path.exists(CHECKPOINT_PATH):
    print(f"\nFound existing checkpoint at {CHECKPOINT_PATH} — resuming from it.\n")
    model = tf.keras.models.load_model(CHECKPOINT_PATH)
else:
    print("\nNo checkpoint found — building a new LSTM model from scratch.\n")
    model = models.Sequential([
        layers.Input(shape=(sequence_length, num_features)),

        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),

        layers.LSTM(32),
        layers.Dropout(0.3),

        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
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
# Step 5: Train
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
# Step 6: Evaluate
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
print("            Predicted Alert  Predicted Yawning")
print(f"Actual Alert       {cm[0][0]:>6}            {cm[0][1]:>6}")
print(f"Actual Yawning     {cm[1][0]:>6}            {cm[1][1]:>6}")

# ---------------------------------------------------------------------------
# Step 7: Save
# ---------------------------------------------------------------------------
final_model_path = os.path.join(MODEL_SAVE_DIR, "lstm_model.keras")
model.save(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")