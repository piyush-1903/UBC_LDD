import hashlib
from pathlib import Path

import cv2
import pandas as pd

repo_dir = Path(__file__).resolve().parents[1]
raw_dir = repo_dir / "data" / "raw" / "born_pocus"
intermediate_dir = repo_dir / "data" / "intermediate"
output_dir = intermediate_dir / "born_pocus"
stills_dir = output_dir / "multi_view_stills"
max_stills = 3

candidates = pd.read_csv(
    intermediate_dir / "baseline_candidates.csv",
    dtype=str,
    keep_default_na=False,
)
candidates = candidates.sort_values("relative_path").drop_duplicates(
    ["source_id", "source_patient_id", "relative_path"]
)
stills_dir.mkdir(parents=True, exist_ok=True)


def video_frame(path, fraction):
    video = cv2.VideoCapture(str(path))
    try:
        if not video.isOpened():
            return None, None

        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        if frame_count <= 0:
            if fraction != 0.5:
                return None, None
            frame_index = 0
        else:
            frame_index = int((frame_count - 1) * fraction)

        if frame_index and not video.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            return None, None

        success, frame = video.read()
        return (frame, frame_index) if success else (None, None)
    finally:
        video.release()


def candidate_frames(recordings):
    paths = [Path(path) for path in recordings["relative_path"]]
    image_paths = [path for path in paths if path.parts[0] == "pocus_images"]
    video_paths = [path for path in paths if path.parts[0] == "pocus_videos"]

    for path in image_paths:
        frame = cv2.imread(str(raw_dir / path))
        if frame is None and path.suffix.lower() == ".gif":
            frame, _ = video_frame(raw_dir / path, 0.0)
        if frame is not None:
            yield path, frame, None, "original_still"

    for fraction in (0.5, 0.25, 0.75):
        for path in video_paths:
            frame, frame_index = video_frame(raw_dir / path, fraction)
            if frame is not None:
                yield path, frame, frame_index, "video_frame"


results = []
patient_columns = ["source_id", "source_patient_id"]
for number, (_, recordings) in enumerate(
    candidates.groupby(patient_columns, sort=True), start=1
):
    group_id = f"born_{number:04d}"
    label = recordings["class"].iloc[0]
    seen_pixels = set()
    used_paths = set()
    saved = 0

    for source_path, frame, frame_index, origin in candidate_frames(recordings):
        fingerprint = (frame.shape, hashlib.sha256(frame.tobytes()).digest())
        if fingerprint in seen_pixels:
            continue

        still_number = saved + 1
        image_path = stills_dir / f"{group_id}_{still_number}.png"
        if not cv2.imwrite(str(image_path), frame):
            raise OSError(f"Could not write {image_path}")

        results.append({
            "group_id": group_id,
            "class": label,
            "still_number": still_number,
            "is_default": still_number == 1,
            "source_path": source_path.as_posix(),
            "frame_index": "" if frame_index is None else frame_index,
            "origin": origin,
            "additional_frame_from_same_clip": source_path in used_paths,
            "image_path": image_path.relative_to(repo_dir).as_posix(),
            "status": "review",
        })
        seen_pixels.add(fingerprint)
        used_paths.add(source_path)
        saved += 1
        if saved == max_stills:
            break

    if saved == 0:
        results.append({
            "group_id": group_id,
            "class": label,
            "still_number": "",
            "is_default": False,
            "source_path": "",
            "frame_index": "",
            "origin": "",
            "additional_frame_from_same_clip": False,
            "image_path": "",
            "status": "no_readable_media",
        })

manifest = pd.DataFrame(results)
manifest_path = output_dir / "multi_view_manifest.csv"
manifest.to_csv(manifest_path, index=False)

group_counts = (
    manifest.assign(saved=manifest["status"].eq("review"))
    .groupby(["class", "group_id"])["saved"]
    .sum()
    .reset_index(name="stills")
)
print("\nPatient groups by number of saved stills:")
print(pd.crosstab(group_counts["class"], group_counts["stills"]).to_string())
print(f"\nSaved {group_counts['stills'].sum()} stills. Manifest: {manifest_path}")