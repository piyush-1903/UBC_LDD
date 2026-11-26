import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from normalize_stills import SIZE


MASK_REFERENCE_SIZE = 224


def box(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


MASKS = {
    "43_litfl_pneumonia": [box(69, 29, 80, 42)],
    "199_paper_covid": [
        box(0, 25, 94, 35), box(64, 35, 72, 42), box(0, 36, 9, 196), box(202, 33, 223, 197),
    ],
    "200_paper_covid": [
        box(0, 25, 94, 35), box(64, 35, 72, 42), box(0, 36, 9, 196), box(202, 33, 223, 197),
    ],
    "217_clarius_normal": [
        box(52, 0, 64, 14), box(41, 28, 56, 45),
        box(176, 24, 180, 28), box(176, 85, 180, 89),
        box(176, 145, 180, 150), box(176, 206, 180, 210), box(153, 214, 176, 223),
    ],
    "158_core_pneumonia": [
        box(31, 155, 75, 168), box(65, 168, 76, 172),
        [(64, 50), (71, 50), (71, 111), (64, 111)],
        [(141, 48), (161, 48), (161, 65), (141, 65)],
    ],
    "170_core_pneumonia": [
        [(95, 44), (105, 44), (105, 54), (95, 54)],
        [(169, 64), (190, 64), (190, 88), (169, 88)],
        box(201, 44, 213, 137), box(203, 138, 213, 179),
        [(34, 162), (54, 162), (54, 178), (34, 178)],
        [(188, 172), (209, 172), (209, 179), (188, 179)],
    ],
}


def save(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Point frames.csv to the masked stills after preview review.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="Redirect preview output for testing.")
    args = parser.parse_args()
    if args.apply and args.output_dir:
        parser.error("--apply uses the repository staging directory")
    root = args.repo_root.resolve()
    stage = root / "data/intermediate/covidx_us/shortlist"
    manifest_path = stage / "frames.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    output = args.output_dir or stage / "overlay_masks"
    for folder in ("previews", "images"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    if "overlay_mask_polygons" not in manifest:
        manifest["overlay_mask_polygons"] = ""

    selected = manifest.loc[manifest["clip_id"].isin(MASKS)]
    if set(selected["clip_id"]) != set(MASKS):
        raise SystemExit("Missing staged clips: " + ", ".join(sorted(set(MASKS) - set(selected["clip_id"]))))

    for idx, row in selected.iterrows():
        name = Path(row["image_path"]).name
        original = cv2.imread(str(stage / "images" / name), cv2.IMREAD_UNCHANGED)
        if original is None or original.shape != (SIZE, SIZE) or original.dtype != np.uint8:
            raise ValueError(f"Expected an original {SIZE} x {SIZE} grayscale PNG for {name}")

        scale = SIZE / MASK_REFERENCE_SIZE
        polygons = [
            [(round(x * scale), round(y * scale)) for x, y in polygon]
            for polygon in MASKS[row["clip_id"]]
        ]
        masked = original.copy()
        for polygon in polygons:
            cv2.fillPoly(masked, [np.asarray(polygon, dtype=np.int32)], 0)
        destination = output / "images" / name
        save(destination, masked)
        save(output / "previews" / name, np.hstack([original, masked]))
        if args.apply:
            manifest.at[idx, "image_path"] = destination.relative_to(root).as_posix()
            manifest.at[idx, "overlay_mask_polygons"] = json.dumps(polygons)

    if args.apply:
        pending = manifest_path.with_suffix(".csv.tmp")
        manifest.to_csv(pending, index=False)
        pending.replace(manifest_path)
    print(f"Masked {len(selected)} frames. Previews in {output / 'previews'}.")


if __name__ == "__main__":
    main()



