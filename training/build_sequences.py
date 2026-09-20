"""
build_sequences.py

Runs the per-frame feature extractor (detection/feature_extraction.py)
across every frame of every YawDD Mirror video, then slices the resulting
per-video feature sequences into fixed-length overlapping windows for
LSTM training.

Labeling strategy:
    - Videos with "Normal" or "Talking" in the filename -> label 0 (Alert)
    - Videos with "Yawning" in the filename            -> label 1 (Drowsy-indicator)

Output:
    training/sequence_data.npz containing:
        X: array of shape (num_sequences, SEQUENCE_LENGTH, 6)
        y: array of shape (num_sequences,)

This is a heavier, slower step than earlier phases since it runs both
CNNs on every processed frame. It supports resuming: already-processed
videos are skipped on a re-run, so you can safely stop (Ctrl+C) and
restart later without losing progress.
"""

import os
import sys
import json
import numpy as np
import cv2

# Add the project root to Python's module search path, so 'detection' can be
# imported even though this script lives inside the training/ subfolder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.feature_extraction import extract_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_FOLDERS = [
    os.path.join("datasets", "YawDD", "YawDD", "YawDD dataset", "Mirror", "Female_mirror"),
    os.path.join("datasets", "YawDD", "YawDD", "YawDD dataset", "Mirror", "Male_mirror Avi Videos"),
]

OUTPUT_DIR = os.path.join("training")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sequence_data.npz")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "sequence_progress.json")

FRAME_SAMPLE_RATE = 6      # process every 6th frame (faster rerun)
SEQUENCE_LENGTH = 20       # frames per sequence window
SEQUENCE_STRIDE = 10       # overlap between windows (10 = 50% overlap)


def classify_video(filename):
    lower_name = filename.lower()
    if "yawning" in lower_name:
        return 1  # Drowsy-indicator
    elif "normal" in lower_name or "talking" in lower_name:
        return 0  # Alert
    else:
        return None  # skip unrecognized files


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"processed_videos": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def load_existing_sequences():
    if os.path.exists(OUTPUT_FILE):
        data = np.load(OUTPUT_FILE)
        return list(data["X"]), list(data["y"])
    return [], []


def extract_video_features(video_path):
    """Runs extract_features() on sampled frames of one video, returns a list of feature vectors."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARNING] Could not open video: {video_path}")
        return []

    features_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SAMPLE_RATE == 0:
            features = extract_features(frame)
            if features is not None:
                features_list.append(features)

        frame_count += 1

    cap.release()
    return features_list


def build_windows(features_list, label):
    """Slices a per-video feature list into fixed-length overlapping windows."""
    windows = []
    labels = []
    for start in range(0, len(features_list) - SEQUENCE_LENGTH + 1, SEQUENCE_STRIDE):
        window = features_list[start:start + SEQUENCE_LENGTH]
        windows.append(window)
        labels.append(label)
    return windows, labels


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    progress = load_progress()
    X_all, y_all = load_existing_sequences()

    total_videos = 0
    processed_this_run = 0
    skipped_files = 0

    video_jobs = []
    for folder in SOURCE_FOLDERS:
        if not os.path.isdir(folder):
            print(f"[WARNING] Source folder not found, skipping: {folder}")
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(".avi"):
                video_jobs.append((folder, fname))

    total_videos = len(video_jobs)
    print(f"Found {total_videos} total videos across source folders.")
    print(f"Already processed in a previous run: {len(progress['processed_videos'])}")
    print("Starting/resuming sequence extraction...\n")

    for folder, fname in video_jobs:
        video_id = os.path.join(folder, fname)

        if video_id in progress["processed_videos"]:
            continue  # already done in a previous run — skip

        label = classify_video(fname)
        if label is None:
            skipped_files += 1
            print(f"  [SKIPPED] {fname} (unrecognized label)")
            progress["processed_videos"].append(video_id)
            continue

        video_path = os.path.join(folder, fname)
        print(f"  Processing: {fname} (label={'Yawning' if label == 1 else 'Alert'}) ...")

        features_list = extract_video_features(video_path)
        windows, labels = build_windows(features_list, label)

        X_all.extend(windows)
        y_all.extend(labels)

        print(f"    -> {len(features_list)} frames processed, {len(windows)} sequence windows created")

        progress["processed_videos"].append(video_id)
        processed_this_run += 1

        # Save progress after every video, so an interruption doesn't lose everything
        save_progress(progress)
        np.savez(
            OUTPUT_FILE,
            X=np.array(X_all, dtype="float32"),
            y=np.array(y_all, dtype="int32"),
        )

    print("\n" + "=" * 60)
    print("Sequence extraction complete (or paused)")
    print("=" * 60)
    print(f"Videos processed this run: {processed_this_run}")
    print(f"Skipped files:             {skipped_files}")
    print(f"Total sequence windows so far: {len(X_all)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nIf you stopped early or this crashed, just run this script again —")
    print("it will skip already-processed videos and continue from where it left off.")


if __name__ == "__main__":
    main()