"""
train_lstm_closure_duration.py

Retrains the LSTM on a properly TEMPORAL task: distinguishing a normal
quick blink from a sustained eye closure (drowsy-indicator) — based on
how many CONSECUTIVE frames the eyes stay closed within each window.

This fixes the earlier data-leakage issue where the label was just an
average of one input feature. Here, the label depends on the ORDER and
RUN-LENGTH of closed-eye frames, which a simple average cannot capture —
this is what actually requires an LSTM instead of a plain CNN average.

No video reprocessing needed: this reuses the per-frame feature vectors
already saved in training/sequence_data.npz from Phase J2.

Feature vector layout (per frame):
    [left_EAR, right_EAR, avg_EAR, MAR, eye_cnn_score, yawn_cnn_score]

Labeling logic:
    - eye_cnn_score < CLOSED_THRESHOLD  -> that frame counts as "closed"
    - Find the LONGEST RUN of consecutive closed frames in the window
    - If longest run >= SUSTAINED_CLOSURE_FRAMES -> label 1 (Drowsy-indicator)
    - Otherwise                                  -> label 0 (Alert / normal blink)

IMPORTANT LIMITATION (documented honestly, not hidden):
YawDD's subjects are alert actors performing Normal/Talking/Yawning —
none of them are genuinely drowsy or falling asleep. So the "Drowsy"
class here is a PROXY for longer closure episodes (e.g. during a slow
blink or the eye-narrowing that often accompanies a yawn), not real
multi-second microsleep events. This is a reasonable proof-of-concept
given available data, but a production system would need real drowsy-
driving footage (e.g. NTHU-DDD) to learn genuine microsleep patterns —
worth stating explicitly in your report's limitations section.
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
MODEL_SAVE_DIR = os.path.join("models", "lstm_model_v3")
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "checkpoint.keras")
BATCH_SIZE = 32
EPOCHS = 25
CLASS_NAMES = ["Alert", "Drowsy-indicator"]

EYE_CNN_SCORE_INDEX = 4  # 0=Closed-leaning, 1=Open-leaning
CLOSED_THRESHOLD = 0.5           # eye_cnn_score below this = "closed" for that frame
SUSTAINED_CLOSURE_FRAMES = 4     # consecutive closed frames needed to count as sustained
                                   # (at FRAME_SAMPLE_RATE=6, 4 sampled frames spans ~24
                                   # raw frames — noticeably longer than a typical blink,
                                   # which usually only shows up in 1 sampled frame)


def longest_closed_run(eye_scores, threshold):
    """Returns the length of the longest consecutive run of 'closed' frames."""
    longest = 0
    current = 0
    for score in eye_scores:
        if score < threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


# ---------------------------------------------------------------------------
# Step 1: Load sequence data and derive duration-based labels
# ---------------------------------------------------------------------------
print("=" * 60)
print("RASD LSTM Training — Sustained Closure Duration Task")
print("=" * 60)
print("Loading existing sequence data...")

data = np.load(SEQUENCE_DATA_PATH)
X = data["X"]

print(f"Total sequences loaded: {len(X)}")

y = np.zeros(len(X), dtype="int32")
run_lengths = []
for i, window in enumerate(X):
    eye_scores = window[:, EYE_CNN_SCORE_INDEX]
    run = longest_closed_run(eye_scores, CLOSED_THRESHOLD)
    run_lengths.append(run)
    y[i] = 1 if run >= SUSTAINED_CLOSURE_FRAMES else 0

run_lengths = np.array(run_lengths)
print(f"Longest-closed-run distribution — min: {run_lengths.min()}, "
      f"max: {run_lengths.max()}, mean: {run_lengths.mean():.2f}")
print(f"Label distribution — Alert: {np.sum(y == 0)}, Drowsy-indicator: {np.sum(y == 1)}")

if np.sum(y == 1) < 20:
    print("\n[WARNING] Very few 'Drowsy-indicator' examples found. Consider lowering "
          "SUSTAINED_CLOSURE_FRAMES if training fails to learn a meaningful pattern.")

# ---------------------------------------------------------------------------
# Step 2: Split into train / validation / test
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# ---------------------------------------------------------------------------
# Step 3: Class weights
# ---------------------------------------------------------------------------
class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train,
)
class_weight_dict = {0: class_weight_values[0], 1: class_weight_values[1]}
print(f"Computed class weights: {class_weight_dict}")

# ---------------------------------------------------------------------------
# Step 4: Build the LSTM (or resume)
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
print("            Predicted Alert  Predicted Drowsy")
print(f"Actual Alert       {cm[0][0]:>6}           {cm[0][1]:>6}")
print(f"Actual Drowsy      {cm[1][0]:>6}           {cm[1][1]:>6}")

# ---------------------------------------------------------------------------
# Step 7: Save
# ---------------------------------------------------------------------------
final_model_path = os.path.join(MODEL_SAVE_DIR, "lstm_model.keras")
model.save(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")