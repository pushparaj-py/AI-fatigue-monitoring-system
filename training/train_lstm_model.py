"""
train_lstm_model.py

Trains an LSTM on the labeled sequence windows built in Phase J2
(training/sequence_data.npz) to classify a temporal sequence of
facial-behavior features as Alert or Yawning (drowsy-indicator).

Each sequence is 20 frames long, with each frame represented by a
6-element feature vector:
    [left_EAR, right_EAR, avg_EAR, MAR, eye_cnn_score, yawn_cnn_score]

This is the temporal-analysis piece of the pipeline — unlike the CNNs,
which classify a single frame, the LSTM looks at how these features
change over the whole sequence to distinguish a normal blink/talk
pattern from a sustained drowsiness-indicating pattern.

Outputs:
- Trained model saved to models/lstm_model/
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEQUENCE_DATA_PATH = os.path.join("training", "sequence_data.npz")
MODEL_SAVE_DIR = os.path.join("models", "lstm_model")
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "checkpoint.keras")
BATCH_SIZE = 32
EPOCHS = 25
CLASS_NAMES = ["Alert", "Yawning"]  # label 0 = Alert, label 1 = Yawning (drowsy-indicator)

# ---------------------------------------------------------------------------
# Step 1: Load the sequence dataset built in Phase J2
# ---------------------------------------------------------------------------
print("=" * 60)
print("RASD LSTM Temporal Drowsiness Training")
print("=" * 60)
print("Loading sequence data...")

if not os.path.exists(SEQUENCE_DATA_PATH):
    raise FileNotFoundError(
        f"{SEQUENCE_DATA_PATH} not found. Run training/build_sequences.py first."
    )

data = np.load(SEQUENCE_DATA_PATH)
X, y = data["X"], data["y"]

print(f"Total sequences loaded: {len(X)}")
print(f"Sequence shape: {X.shape[1:]} (frames per sequence, features per frame)")
print(f"Class distribution — Alert: {np.sum(y == 0)}, Yawning: {np.sum(y == 1)}")

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
# Step 3: Compute class weights to handle any imbalance
# ---------------------------------------------------------------------------
class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train,
)
class_weight_dict = {0: class_weight_values[0], 1: class_weight_values[1]}
print(f"Computed class weights: {class_weight_dict}")

# ---------------------------------------------------------------------------
# Step 4: Build the LSTM (or resume from a saved checkpoint, if one exists)
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
        layers.Dense(1, activation="sigmoid"),  # binary output: 0=Alert, 1=Yawning
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
print("            Predicted Alert  Predicted Yawning")
print(f"Actual Alert       {cm[0][0]:>6}            {cm[0][1]:>6}")
print(f"Actual Yawning     {cm[1][0]:>6}            {cm[1][1]:>6}")

# ---------------------------------------------------------------------------
# Step 7: Save the final trained model
# ---------------------------------------------------------------------------
final_model_path = os.path.join(MODEL_SAVE_DIR, "lstm_model.keras")
model.save(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")
print(f"(Epoch checkpoint remains at: {CHECKPOINT_PATH})")