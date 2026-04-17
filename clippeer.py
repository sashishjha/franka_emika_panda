import cv2
import numpy as np
from pathlib import Path

# Path to your video file
desktop = Path("/Users/sashishjha/Desktop")
matches = sorted(desktop.glob("Screen Recording 2026-04-17 at 11.56.42*PM.mov"))
if not matches:
    raise FileNotFoundError("Could not find the screen recording on the Desktop")
video_path = str(matches[0])

# Folder where extracted frames will be saved
output_root = desktop / "frames"
output_root.mkdir(parents=True, exist_ok=True)

requested_counts = [100, 50, 10]
output_folders = {count: output_root / str(count) for count in requested_counts}
for folder in output_folders.values():
    folder.mkdir(parents=True, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {video_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

def sample_indices(source_indices, count):
    if count <= 0:
        return []
    if count >= len(source_indices):
        return list(source_indices)
    positions = np.linspace(0, len(source_indices) - 1, num=count, dtype=int)
    return [source_indices[i] for i in positions]

if total_frames <= 0:
    raise RuntimeError("Could not determine total frame count for the video")

base_count = min(requested_counts[0], total_frames)
base_indices = sample_indices(range(total_frames), base_count)

indices_by_count = {
    100: base_indices,
    50: sample_indices(base_indices, min(50, len(base_indices))),
    10: sample_indices(base_indices, min(10, len(base_indices))),
}

save_map = {}
for count, indices in indices_by_count.items():
    for index in indices:
        save_map.setdefault(index, []).append(count)

print(f"Video frames: {total_frames}, FPS: {fps:.3f}")
print(f"Saving {len(indices_by_count[100])} frames to {output_folders[100]}")
print(f"Saving {len(indices_by_count[50])} frames to {output_folders[50]}")
print(f"Saving {len(indices_by_count[10])} frames to {output_folders[10]}")

frame_count = 0
saved_counts = {count: 0 for count in requested_counts}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count in save_map:
        timestamp_ms = int(round((frame_count / fps) * 1000)) if fps > 0 else frame_count
        filename = f"frame_{frame_count:06d}_t{timestamp_ms:010d}ms.jpg"
        for count in save_map[frame_count]:
            output_path = output_folders[count] / filename
            cv2.imwrite(str(output_path), frame)
            print(f"Saved {output_path}")
            saved_counts[count] += 1

    frame_count += 1

cap.release()

print("\nDone.")
for count in requested_counts:
    print(f"Saved {saved_counts[count]} frames to '{output_folders[count]}'")