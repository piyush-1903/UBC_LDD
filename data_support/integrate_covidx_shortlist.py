import argparse
import shutil
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import pandas as pd

from normalize_stills import SIZE, fit_to_canvas

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/covidx_us"
STAGE = ROOT / "data/intermediate/covidx_us/shortlist"
CLEAN = ROOT / "data/clean"
CLIPS = {
    "43_litfl_pneumonia": ("litfl_43", "pneumonia"),
    "158_core_pneumonia": ("core_158", "pneumonia"),
    "170_core_pneumonia": ("core_170", "pneumonia"),
    "174_core_covid": ("core_174", "covid"),
    "199_paper_covid": ("kalafat", "covid"),
    "200_paper_covid": ("kalafat", "covid"),
    "217_clarius_normal": ("clarius_217", "healthy"),
}
CLARIUS_URL = "https://clarius.com/wp-content/uploads/2022/01/Clarius_lung-clip-c3-cropped.mp4"


def read_csv(path, **kwargs):
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    STAGE.mkdir(parents=True, exist_ok=True)
    manifest_path = STAGE / "frames.csv"
    if args.publish:
        publish(manifest_path)
        return

    metadata = read_csv(
        RAW / "video_metadata.csv", encoding="cp1252"
    ).set_index("id")
    videos = RAW / "videos"
    videos.mkdir(exist_ok=True)
    images = STAGE / "images"
    images.mkdir(exist_ok=True)

    rows = []
    for clip, (group, label) in CLIPS.items():
        meta = metadata.loc[clip]
        path = videos / f"{clip}.{meta['filetype']}"
        url = CLARIUS_URL if clip == "217_clarius_normal" else meta["url"]

        if not path.exists() and args.download and "vimeo.com" not in url:
            partial = path.with_name(f"{path.stem}.part{path.suffix}")
            try:
                urlretrieve(url, partial)
                downloaded = cv2.VideoCapture(str(partial))
                try:
                    readable, _ = downloaded.read()
                finally:
                    downloaded.release()
                if not readable:
                    raise OSError("Response is not decodable video")
                partial.replace(path)
            except (OSError, ValueError) as exc:
                partial.unlink(missing_ok=True)
                print(f"{clip}: download failed ({exc})")

        if not path.exists():
            print(f"{clip}: obtain from source and save as {path.name}")
            continue

        video = cv2.VideoCapture(str(path))
        try:
            count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if count < 1:
                print(f"{clip}: unreadable video")
                continue

            indices = dict.fromkeys(
                int((count - 1) * fraction)
                for fraction in (0.50, 0.25, 0.75)
            )
            for index in indices:
                if index and not video.set(cv2.CAP_PROP_POS_FRAMES, index):
                    print(f"{clip}: could not seek to frame {index}")
                    continue
                ok, frame = video.read()
                if not ok:
                    print(f"{clip}: could not read frame {index}")
                    continue

                h, w = frame.shape[:2]
                x0, y0, x1, y1 = 0, 0, w, h
                crop_status = "needs_crop_review"

                if clip == "174_core_covid":
                    if (w, h) != (1920, 1080):
                        raise ValueError(
                            f"Unexpected frame size for {clip}: {w} x {h}"
                        )
                    x0, y0, x1, y1 = 64, 110, 700, 850
                    crop_status = "review_crop"

                gray = cv2.cvtColor(
                    frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY
                )
                dest = images / f"covidx_us_{clip}_frame{index:04d}.png"
                if not cv2.imwrite(str(dest), fit_to_canvas(gray, SIZE)):
                    raise OSError(f"Could not write {dest}")

                rows.append({
                    "group_id": f"covidx_us_{group}",
                    "dataset": "covidx_us",
                    "group_scope": "source_patient",
                    "clip_id": clip,
                    "class": label,
                    "source_id": meta["source"],
                    "source_patient_id": meta["patient"],
                    "source_url": url,
                    "source_path": path.relative_to(RAW).as_posix(),
                    "probe": meta["probe"],
                    "origin": "video_frame",
                    "frame_index": index,
                    "image_path": dest.relative_to(ROOT).as_posix(),
                    "original_width": w,
                    "original_height": h,
                    "crop_x0": x0,
                    "crop_y0": y0,
                    "crop_x1": x1,
                    "crop_y1": y1,
                    "output_width": SIZE,
                    "output_height": SIZE,
                    "normalization_qc": crop_status,
                    "status": "review",
                })
        finally:
            video.release()

    if not rows:
        raise SystemExit("No readable frames staged.")

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Staged {len(rows)} frames under {STAGE}")

    
def publish(manifest_path):
    staged = read_csv(manifest_path)
    approvals = read_csv(STAGE / "approved.csv")
    if approvals["image_path"].duplicated().any():
        raise SystemExit("Duplicate image paths in approved.csv")
    if not set(approvals["image_path"]).issubset(staged["image_path"]):
        raise SystemExit("Approval references an image outside the staged manifest.")
    approved = staged.merge(approvals[["image_path", "reuse_evidence"]], on="image_path")
    if approved.empty:
        raise SystemExit("No approved stills to publish.")
    approved["still_number"] = approved.groupby("group_id").cumcount() + 1
    if approved["still_number"].gt(3).any():
        raise SystemExit("Approve at most three stills per patient group.")
    approved["is_default"] = approved["still_number"].eq(1)
    approved["additional_frame_from_same_clip"] = approved.duplicated(["group_id", "clip_id"])
    approved["normalization_qc"] = "approved"
    approved["status"] = "approved"

    combined_path = CLEAN / "manifests/normalization_manifest.csv"
    current = read_csv(combined_path)
    born = current.loc[current["dataset"].eq("born_pocus")]
    if born.empty:
        raise SystemExit("The shared manifest must contain the Born POCUS baseline.")
    for label, rows in (("Born POCUS", born), ("approved COVIDx-US", approved)):
        if not rows["output_width"].eq(str(SIZE)).all() or not rows["output_height"].eq(str(SIZE)).all():
            raise SystemExit(f"Regenerate {label} images at {SIZE} x {SIZE} before publishing.")
    old_covidx = current.loc[current["dataset"].eq("covidx_us"), "normalized_path"]
    names = approved["image_path"].map(lambda p: Path(p).name)
    approved["normalized_path"] = "data/clean/images/" + names
    combined = pd.concat([born, approved], ignore_index=True).fillna("")
    if combined["normalized_path"].duplicated().any():
        raise SystemExit("Output path collision.")
    for row in approved.itertuples():
        shutil.copyfile(ROOT / row.image_path, ROOT / row.normalized_path)
    partial = combined_path.with_suffix(".csv.tmp")
    combined.to_csv(partial, index=False)
    partial.replace(combined_path)
    archive = STAGE / "previous_export"
    archive.mkdir(exist_ok=True)
    for path in set(old_covidx) - set(approved["normalized_path"]):
        source = ROOT / path
        if source.parent == CLEAN / "images" and source.is_file():
            target = archive / source.name
            if target.exists():
                raise OSError(f"Archive destination already exists: {target}")
            source.rename(target)
    print(f"Published {len(approved)} COVIDx-US stills across {approved['group_id'].nunique()} groups.")


if __name__ == "__main__":
    main()

