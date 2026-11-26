import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
INTERMEDIATE = DATA / "intermediate"
BORN = INTERMEDIATE / "born_pocus"
COVIDX = INTERMEDIATE / "covidx_us"
CLEAN = DATA / "clean"
RULES_DIR = REPO / "data_support"
SIZE = 512


def fit_to_canvas(image, size):
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=interpolation
    )

    canvas = np.zeros((size, size, *image.shape[2:]), dtype=image.dtype)
    x = (size - new_width) // 2
    y = (size - new_height) // 2
    canvas[y:y + new_height, x:x + new_width] = resized
    return canvas


def choose_pilot(images, count=12):
    # Start with one default view per patient group.
    defaults = images.loc[images["is_default"].str.casefold().eq("true")]
    defaults = defaults.sort_values(["class", "probe", "origin", "image_path"])

    batches = []
    for _, class_rows in defaults.groupby("class", sort=True):
        # Put different probe and image origins first within each class.
        first = class_rows.drop_duplicates(["probe", "origin"])
        remaining = class_rows.drop(index=first.index)
        batches.append(pd.concat([first, remaining]).head(count // 3))

    selected = pd.concat(batches)
    remaining = defaults.drop(index=selected.index)
    return pd.concat([selected, remaining]).head(count)


def write_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def load_covidx_stills():
    drafts = pd.read_csv(
        COVIDX / "draft_stills_manifest.csv", dtype=str, keep_default_na=False
    )
    metadata = pd.read_csv(
        DATA / "raw/covidx_us/video_metadata.csv",
        encoding="cp1252",
        dtype=str,
        keep_default_na=False,
    )
    drafts = drafts.loc[drafts["status"].eq("review")].merge(
        metadata[["id", "probe"]], left_on="clip_id", right_on="id",
        validate="many_to_one", how="left", indicator=True,
    )
    if drafts["_merge"].ne("both").any():
        raise SystemExit("Some COVIDx-US stills have no matching metadata.")
    drafts = drafts.drop(columns=["_merge", "id"])
    has_patient = drafts["source_patient_id"].str.strip().ne("")
    patient_key = drafts["source"].str.casefold() + "_" + drafts["source_patient_id"]
    drafts["group_id"] = ("covidx_us_" + patient_key).where(
        has_patient, "covidx_us_clip_" + drafts["clip_id"]
    )
    drafts["group_scope"] = np.where(has_patient, "source_patient", "clip_only")
    drafts["still_number"] = drafts.groupby("group_id").cumcount() + 1
    drafts["is_default"] = drafts["still_number"].eq(1).map({True: "True", False: "False"})
    drafts["additional_frame_from_same_clip"] = drafts.duplicated(["group_id", "clip_id"])
    drafts["source_id"] = drafts.pop("source")
    drafts["origin"] = "video_frame"
    drafts["dataset"] = "covidx_us"
    return drafts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Process 12 sample stills and save previews.")
    parser.add_argument(
        "--include-covidx-drafts", action="store_true",
        help="Include COVIDx-US review stills in the combined draft export.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(
        BORN / "multi_view_manifest.csv", dtype=str, keep_default_na=False
    )
    candidates_path = BORN / "baseline_candidates.csv"
    if not candidates_path.is_file():
        candidates_path = INTERMEDIATE / "baseline_candidates.csv"
    candidates = pd.read_csv(candidates_path, dtype=str, keep_default_na=False)
    provenance = candidates[["relative_path", "source_id", "probe"]].drop_duplicates()
    images = manifest.loc[manifest["status"].eq("review")].merge(
        provenance,
        left_on="source_path",
        right_on="relative_path",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if images.empty:
        raise SystemExit("The multi-view manifest has no stills ready for review.")
    if images["_merge"].ne("both").any():
        raise SystemExit("Some stills have no matching source in baseline_candidates.csv.")

    images = images.drop(columns=["_merge", "relative_path"])
    images["dataset"] = "born_pocus"
    images["group_scope"] = "source_patient"
    if args.include_covidx_drafts:
        images = pd.concat([images, load_covidx_stills()], ignore_index=True).fillna("")

    rules = {}
    for dataset in images["dataset"].unique():
        path = RULES_DIR / f"{dataset}_crops.json"
        rules[dataset] = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}

    if args.pilot:
        selected = choose_pilot(images.loc[images["dataset"].eq("born_pocus")])
        if args.include_covidx_drafts:
            covidx = images.loc[
                images["dataset"].eq("covidx_us") & images["is_default"].eq("True")
            ]
            selected = pd.concat([selected, covidx], ignore_index=True)
    else:
        selected = images

    output = INTERMEDIATE / "normalization_pilot" if args.pilot else CLEAN
    filenames = [
        Path(path).name if dataset == "born_pocus" else f"covidx_us_{Path(path).name}"
        for dataset, path in zip(selected["dataset"], selected["image_path"])
    ]
    if len({name.casefold() for name in filenames}) != len(filenames):
        raise SystemExit("Multiple input stills map to the same output filename.")
    selected = selected.assign(output_filename=filenames)
    png_dir = output / "images"
    png_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output / "previews"
    if args.pilot:
        preview_dir.mkdir(exist_ok=True)

    results = []
    for row in selected.to_dict("records"):
        path = REPO / row["image_path"]
        # Keep the colour original for the before-and-after preview.
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if original is None:
            raise OSError(f"Could not read {path}")

        height, width = original.shape[:2]
        crop = rules[row["dataset"]].get(row["source_path"])
        x0, y0, x1, y1 = crop if crop is not None else (0, 0, width, height)

        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(
                f"Crop {crop} is outside {width}x{height} for {row['source_path']}"
            )

        grayscale = cv2.cvtColor(original[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        normalized = fit_to_canvas(grayscale, SIZE)
        filename = row.pop("output_filename")
        normalized_path = png_dir / filename
        write_png(normalized_path, normalized)

        if args.pilot:
            before = fit_to_canvas(original, SIZE)
            after = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            preview = np.hstack([before, after])
            write_png(preview_dir / filename, preview)

        results.append({
            **row,
            "original_width": width,
            "original_height": height,
            "crop_x0": x0,
            "crop_y0": y0,
            "crop_x1": x1,
            "crop_y1": y1,
            "output_width": SIZE,
            "output_height": SIZE,
            "normalized_path": normalized_path.relative_to(REPO).as_posix(),
            "normalization_qc": "review_crop" if crop is not None else "needs_crop_review",
        })

    manifest_dir = output / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(manifest_dir / "normalization_manifest.csv", index=False)
    print(f"Processed {len(results)} stills. Output: {output.relative_to(REPO)}")


if __name__ == "__main__":
    main()