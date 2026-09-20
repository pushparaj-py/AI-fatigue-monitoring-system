"""
extract_yawn_frames.py

Preprocessing script for the YawDD Mirror dataset.

The Mirror dataset's videos are already labeled by filename:
    <id>-<description>-Normal.avi
    <id>-<description>-Talking.avi
    <id>-<description>-Yawning.avi

This script extracts frames from these videos and sorts them into
two output folders based on filename:
    datasets/YawDD_frames/Yawning/    <- frames from *-Yawning.avi
    datasets/YawDD_frames/NotYawning/ <- frames from *-Normal.avi and *-Talking.avi

These folders can then be used directly for CNN training, the same way
MRL's Open/Closed folders were used for the eye model.
"""

import os
import cv2

# Source folders (Mirror dataset — pre-labeled by filename)
SOURCE_FOLDERS = [
    os.path.join("datasets", "YawDD", "YawDD", "YawDD dataset", "Mirror", "Female_mirror"),
    os.path.join("datasets", "YawDD", "YawDD", "YawDD dataset", "Mirror", "Male_mirror Avi Videos"),
]

# Output folders for extracted frames
OUTPUT_DIR = os.path.join("datasets", "YawDD_frames")
YAWNING_DIR = os.path.join(OUTPUT_DIR, "Yawning")
NOT_YAWNING_DIR = os.path.join(OUTPUT_DIR, "NotYawning")

# Extract every Nth frame instead of every single frame, to avoid
# generating hundreds of thousands of near-duplicate images from
# a handful of videos. Adjust if you want more/fewer frames.
FRAME_SAMPLE_RATE = 5


def classify_video(filename):
    """
    Returns 'Yawning', 'NotYawning', or None (skip) based on filename.
    Only Normal/Talking/Yawning videos are used — anything else is skipped.
    """
    lower_name = filename.lower()
    if "yawning" in lower_name:
        return "Yawning"
    elif "normal" in lower_name or "talking" in lower_name:
        return "NotYawning"
    else:
        return None  # skip unrecognized files


def extract_frames_from_video(video_path, output_folder, video_label):
    """Extracts every Nth frame from a video and saves it as a labeled image."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARNING] Could not open video: {video_path}")
        return 0

    frame_count = 0
    saved_count = 0
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SAMPLE_RATE == 0:
            out_filename = f"{base_name}_frame{frame_count}.jpg"
            out_path = os.path.join(output_folder, out_filename)
            cv2.imwrite(out_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count


def main():
    os.makedirs(YAWNING_DIR, exist_ok=True)
    os.makedirs(NOT_YAWNING_DIR, exist_ok=True)

    total_yawning_frames = 0
    total_not_yawning_frames = 0
    skipped_files = 0

    for folder in SOURCE_FOLDERS:
        if not os.path.isdir(folder):
            print(f"[WARNING] Source folder not found, skipping: {folder}")
            continue

        video_files = [f for f in os.listdir(folder) if f.lower().endswith(".avi")]
        print(f"\nProcessing {len(video_files)} videos in: {folder}")

        for video_file in video_files:
            label = classify_video(video_file)
            video_path = os.path.join(folder, video_file)

            if label == "Yawning":
                count = extract_frames_from_video(video_path, YAWNING_DIR, label)
                total_yawning_frames += count
                print(f"  [Yawning]    {video_file} -> {count} frames")
            elif label == "NotYawning":
                count = extract_frames_from_video(video_path, NOT_YAWNING_DIR, label)
                total_not_yawning_frames += count
                print(f"  [NotYawning] {video_file} -> {count} frames")
            else:
                skipped_files += 1
                print(f"  [SKIPPED]    {video_file} (unrecognized label)")

    print("\n" + "=" * 60)
    print("Frame extraction complete")
    print("=" * 60)
    print(f"Total Yawning frames:     {total_yawning_frames}")
    print(f"Total NotYawning frames:  {total_not_yawning_frames}")
    print(f"Skipped files:            {skipped_files}")
    print(f"\nOutput folders:")
    print(f"  {YAWNING_DIR}")
    print(f"  {NOT_YAWNING_DIR}")


if __name__ == "__main__":
    main()