import hashlib
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/covidx_us"
OUT = ROOT / "data/intermediate/covidx_us"

CLIPS = (
    "43_litfl_pneumonia.mp4",
    "170_core_pneumonia.mp4",
    "65_pocusatlas_pneumonia.gif",
    "68_pocusatlas_pneumonia.gif",
    "70_pocusatlas_pneumonia.gif",
    "75_pocusatlas_pneumonia.gif",
)

metadata = pd.read_csv(
    RAW / "video_metadata.csv", encoding="cp1252", dtype=str, keep_default_na=False
)
metadata = metadata.set_index("id", verify_integrity=True)

stills_dir = OUT / "stills"
stills_dir.mkdir(parents=True, exist_ok=True)
records = []

for filename in CLIPS:
    path = RAW / "videos" / filename
    source = metadata.loc[path.stem]
    video = cv2.VideoCapture(str(path))
    saved = 0
    seen = set()
    try:
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            raise OSError(f"Could not read frame count for {path.name}")

        for fraction in (0.50, 0.25, 0.75):
            index = int((frame_count - 1) * fraction)
            if not video.set(cv2.CAP_PROP_POS_FRAMES, index):
                continue
            ok, frame = video.read()
            if not ok:
                continue

            digest = hashlib.sha256(frame.tobytes()).digest()
            if digest in seen:
                continue

            dest = stills_dir / f"{path.stem}_frame{index:04d}.png"
            if not cv2.imwrite(str(dest), frame):
                raise OSError(f"Could not save {dest}")

            records.append({
                "clip_id": path.stem,
                "class": "pneumonia",
                "source": source["source"],
                "source_patient_id": source["patient"],
                "source_path": path.relative_to(ROOT).as_posix(),
                "frame_index": index,
                "image_path": dest.relative_to(ROOT).as_posix(),
                "status": "review",
            })
            seen.add(digest)
            saved += 1
    finally:
        video.release()

    print(f"{path.name}: {saved} draft stills")
    if saved == 0:
        raise OSError(f"No frames could be extracted from {path.name}")

pd.DataFrame(records).to_csv(OUT / "draft_stills_manifest.csv", index=False)
print(f"Saved {len(records)} draft stills from {len(CLIPS)} clips.")