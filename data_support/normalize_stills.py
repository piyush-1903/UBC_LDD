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
CLEAN = DATA / "clean" / "born_pocus"
CROP_RULES = REPO / "data_support" / "born_pocus_crops.json"
SIZE = 224


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Process 12 sample stills and save previews.")
    args = parser.parse_args()

    manifest = pd.read_csv(
        BORN / "multi_view_manifest.csv", dtype=str, keep_default_na=False
    )
    candidates = pd.read_csv(
        INTERMEDIATE / "baseline_candidates.csv", dtype=str, keep_default_na=False
    )
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

    rules = {}
    if CROP_RULES.exists():
        rules = json.loads(CROP_RULES.read_text(encoding="utf-8"))

    selected = choose_pilot(images) if args.pilot else images
    output = BORN / "normalization_pilot" if args.pilot else CLEAN
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
        crop = rules.get(row["source_path"])
        x0, y0, x1, y1 = crop if crop is not None else (0, 0, width, height)

        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(
                f"Crop {crop} is outside {width}x{height} for {row['source_path']}"
            )

        grayscale = cv2.cvtColor(original[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        normalized = fit_to_canvas(grayscale, SIZE)
        normalized_path = png_dir / path.name
        write_png(normalized_path, normalized)

        if args.pilot:
            before = fit_to_canvas(original, SIZE)
            after = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            preview = np.hstack([before, after])
            write_png(preview_dir / path.name, preview)

        row.pop("_merge")
        row.pop("relative_path")
        results.append({
            **row,
            "dataset": "born_pocus",
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

    pd.DataFrame(results).to_csv(output / "normalization_manifest.csv", index=False)
    print(f"Processed {len(results)} stills. Output: {output.relative_to(REPO)}")


if __name__ == "__main__":
    main()