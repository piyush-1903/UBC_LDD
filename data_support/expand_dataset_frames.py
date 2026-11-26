import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import mkdtemp

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageSequence

from build_katumba_manifest import normalized_key, workbook_rows
from mask_covidx_overlays import MASKS, MASK_REFERENCE_SIZE
from normalize_stills import SIZE, fit_to_canvas


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mpeg", ".mpg", ".m4v", ".webm", ".wmv"}
RESTRICTED_CLIP = "174_core_covid"
DUPLICATE_KATUMBA_GROUP = (
    "katumba_25b076033cd9f4a5cb9fd3b68a23197386ef613197e71a8f94f761d762f7ef14"
)
DATASETS = {"born_pocus", "covidx_us", "katumba"}


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def decoded_frames(path, allowed=None):
    suffix = path.suffix.lower()
    if suffix == ".gif":
        with Image.open(path) as source:
            animated = source.is_animated
            for index, frame in enumerate(ImageSequence.Iterator(source)):
                if allowed is None or index in allowed:
                    rgb = np.asarray(frame.convert("RGB"))
                    yield index if animated else None, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return

    if suffix not in VIDEO_EXTENSIONS:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not decode image: {path}")
        yield None, image
        return

    video = cv2.VideoCapture(str(path))
    try:
        index = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if allowed is None or index in allowed:
                yield index, frame
            index += 1
        if index == 0:
            raise OSError(f"Could not decode video: {path}")
        if allowed and max(allowed) >= index:
            raise ValueError(f"Selected frame lies outside decoded video: {path}")
    finally:
        video.release()


def born_frames(root, current):
    intermediate = root / "data/intermediate"
    candidates = intermediate / "born_pocus/baseline_candidates.csv"
    if not candidates.exists():
        candidates = intermediate / "baseline_candidates.csv"
    _, rows = read_rows(candidates)
    records = defaultdict(dict)
    for row in rows:
        key = row["source_id"], row["source_patient_id"]
        records[key][row["relative_path"]] = row

    rules_path = root / "data_support/born_pocus_crops.json"
    rules = json.loads(rules_path.read_text(encoding="utf-8-sig")) if rules_path.exists() else {}
    known_groups = {row["source_path"]: row["group_id"] for row in current}
    for number, key in enumerate(sorted(records), start=1):
        group = f"born_{number:04d}"
        paths = records[key]
        if len({row["class"] for row in paths.values()}) != 1:
            raise ValueError(f"Conflicting Born POCUS labels for {group}")
        for relative_path in sorted(paths, key=lambda path: (not path.startswith("pocus_images/"), path)):
            if relative_path in known_groups and known_groups[relative_path] != group:
                raise ValueError(f"Born POCUS grouping differs from the published export: {relative_path}")
            row = paths[relative_path]
            meta = {
                **row,
                "dataset": "born_pocus",
                "group_id": group,
                "group_scope": "source_patient",
                "source_path": relative_path,
                "status": "review",
            }
            path = root / "data/raw/born_pocus" / relative_path
            yield from normalized_frames(path, meta, rules.get(relative_path))


def normalized_frames(path, meta, crop=None, allowed=None, approved_indices=()):
    polygons = [
        [(round(x * SIZE / MASK_REFERENCE_SIZE), round(y * SIZE / MASK_REFERENCE_SIZE))
         for x, y in polygon]
        for polygon in MASKS.get(meta.get("clip_id"), ())
    ]
    for index, frame in decoded_frames(path, allowed):
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = crop or (0, 0, width, height)
        if meta.get("clip_id") == RESTRICTED_CLIP and (width, height) != (1920, 1080):
            raise ValueError(f"Unexpected dimensions for restricted clip: {path}")
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Invalid crop for {path}: {(x0, y0, x1, y1)}")
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        image = fit_to_canvas(gray, SIZE)
        for polygon in polygons:
            cv2.fillPoly(image, [np.asarray(polygon, dtype=np.int32)], 0)

        qc = "review_crop" if crop else "needs_crop_review"
        if meta["dataset"] == "covidx_us":
            qc = "approved" if index in approved_indices else "needs_frame_review"
        yield {
            **meta,
            "frame_index": index if index is not None else "",
            "origin": "original_still" if index is None else "video_frame",
            "original_width": width,
            "original_height": height,
            "crop_x0": x0,
            "crop_y0": y0,
            "crop_x1": x1,
            "crop_y1": y1,
            "overlay_mask_polygons": json.dumps(polygons) if polygons else "",
            "normalization_qc": qc,
        }, image


def covidx_frames(root, current, lung_range):
    clips = defaultdict(list)
    for row in current:
        if row["status"] == "approved":
            clips[row["clip_id"]].append(row)
    if not clips or "" in clips:
        raise ValueError("Expected an approved COVIDx-US shortlist in the current clean manifest.")

    for clip, rows in sorted(clips.items()):
        meta = rows[0].copy()
        if len({(r["group_id"], r["class"], r["source_path"], r["reuse_evidence"]) for r in rows}) != 1:
            raise ValueError(f"Inconsistent clip metadata for {clip}")
        approved = {int(r["frame_index"]) for r in rows if r["normalization_qc"] == "approved"}
        # Reuse the verified crop from the published clip.
        crop = tuple(int(meta[f"crop_{edge}"]) for edge in ("x0", "y0", "x1", "y1"))
        if crop == (0, 0, int(meta["original_width"]), int(meta["original_height"])):
            crop = None
        allowed = None
        if clip == RESTRICTED_CLIP:
            if lung_range is None:
                allowed = approved
                if not allowed:
                    raise ValueError("Clip 174 needs approved frames or --lung-range-174.")
                print(f"{clip}: using only previously approved frames {sorted(allowed)}")
            else:
                first, last = lung_range
                if not 0 <= first <= last:
                    raise ValueError("The verified frame range must be nonnegative and inclusive.")
                allowed = range(first, last + 1)
        path = root / "data/raw/covidx_us" / meta["source_path"]
        yield from normalized_frames(path, meta, crop, allowed, approved)


def katumba_frames(root, current):
    raw = root / "data/raw/katumba"
    workbook = workbook_rows(raw / "metadata.xlsx")
    fields = {name: col for col, name in next(workbook).items()}
    covid_groups = set()
    for row in workbook:
        patient = normalized_key(row.get(fields["Patient Number"], ""))
        label = row.get(fields["Conclusion"], "").casefold()
        symptoms = row.get(fields["Does the patient have any symptoms"], "")
        if patient and label == "probably covid" and re.search(r"\bcovid\s+rdt\s+positive\b", symptoms, re.I):
            covid_groups.add("katumba_" + hashlib.sha256(patient.encode()).hexdigest())
    published_groups = {row["group_id"] for row in current if row["class"] == "covid"}
    covid_groups = (covid_groups & published_groups) - {DUPLICATE_KATUMBA_GROUP}
    if len(covid_groups) != 4:
        raise ValueError("Expected the four reviewed, explicitly RDT-positive Katumba COVID groups.")

    _, intake = read_rows(root / "data/intermediate/katumba/intake_manifest.csv")
    for row in sorted(intake, key=lambda item: item["source_path"].casefold()):
        if row["status"] != "candidate":
            continue
        if row["class"] != "healthy" and row["group_id"] not in covid_groups:
            continue
        path = root / row["source_path"]
        with Image.open(path) as source:
            gray = ImageOps.exif_transpose(source).convert("L")
            width, height = gray.size
            image = ImageOps.pad(gray, (SIZE, SIZE), method=Image.Resampling.LANCZOS, color=0)
        yield {
            **row,
            "source_path": path.relative_to(raw).as_posix(),
            "source_id": path.stem,
            "group_scope": "source_patient",
            "frame_index": "",
            "origin": "original_still",
            "original_width": width,
            "original_height": height,
            "crop_x0": 0,
            "crop_y0": 0,
            "crop_x1": width,
            "crop_y1": height,
            "normalization_qc": "needs_crop_review",
            "status": "approved",
        }, np.asarray(image)


def build(root, target, image_prefix, fieldnames, current, lung_range):
    image_dir = target / "images"
    image_dir.mkdir()
    manifest_dir = target / "manifests"
    manifest_dir.mkdir()
    rows_by_source = defaultdict(list)
    for row in current:
        rows_by_source[row["dataset"]].append(row)
    frame_sets = (
        born_frames(root, rows_by_source["born_pocus"]),
        covidx_frames(root, rows_by_source["covidx_us"], lung_range),
        katumba_frames(root, rows_by_source["katumba"]),
    )
    owners = {}
    duplicates = []
    group_counts = Counter()
    sources_seen = set()
    results = []
    for dataset_name, frames in zip(("born_pocus", "covidx_us", "katumba"), frame_sets):
        print(f"Processing {dataset_name}...", flush=True)
        for row, pixels in frames:
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            dataset, group = row["dataset"], row["group_id"]
            if digest in owners:
                kept = owners[digest]
                duplicates.append({
                    "dataset": dataset, "group_id": group, "class": row["class"],
                    "source_path": row["source_path"], "frame_index": row["frame_index"],
                    "kept_dataset": kept["dataset"], "kept_group_id": kept["group_id"],
                    "kept_class": kept["class"], "kept_path": kept["normalized_path"],
                })
                continue
            group_counts[group] += 1
            number = group_counts[group]
            filename = f"{group}_{number:05d}.png"
            if not cv2.imwrite(str(image_dir / filename), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"Could not write {filename}")
            path = f"{image_prefix}/{filename}"
            recording = (dataset, group, row["source_path"])
            row.update(
                output_width=SIZE, output_height=SIZE, still_number=number,
                is_default=number == 1, normalized_path=path, image_path="",
                additional_frame_from_same_clip=row["origin"] == "video_frame" and recording in sources_seen,
            )
            sources_seen.add(recording)
            owners[digest] = row
            results.append(row)

    expected_groups = {r["group_id"] for r in current if r["dataset"] != "katumba" or r["class"] == "healthy"}
    missing = expected_groups - group_counts.keys()
    if missing:
        raise ValueError(f"Expected published groups missing after decoding or deduplication: {sorted(missing)}")
    covid_group_count = len({r["group_id"] for r in results if r["dataset"] == "katumba" and r["class"] == "covid"})
    if covid_group_count != 4:
        raise ValueError("Expanded export did not retain all four eligible Katumba COVID groups.")
    write_rows(manifest_dir / "normalization_manifest.csv", fieldnames, results)
    write_rows(manifest_dir / "exact_duplicates.csv",
               ["dataset", "group_id", "class", "source_path", "frame_index",
                "kept_dataset", "kept_group_id", "kept_class", "kept_path"], duplicates)
    for dataset in sorted(DATASETS):
        source_rows = [r for r in results if r["dataset"] == dataset]
        print(f"{dataset}: {len(source_rows)} stills, {len({r['group_id'] for r in source_rows})} groups")
    print(f"Total: {len(results)} stills, {len(group_counts)} groups")
    print(f"Skipped {len(duplicates)} exact normalized duplicates.")
    print(f"Cross-group duplicate pairs: {sum(r['group_id'] != r['kept_group_id'] for r in duplicates)}")
    print(f"Cross-label duplicate pairs: {sum(r['class'] != r['kept_class'] for r in duplicates)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--lung-range-174", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        help="Verified inclusive zero-based lung-only frame range for clip 174.")
    parser.add_argument("--publish", action="store_true",
                        help="Replace data/clean with the expanded export and archive the prior release.")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    clean = root / "data/clean"
    fields, current = read_rows(clean / "manifests/normalization_manifest.csv")
    if {r["dataset"] for r in current} != DATASETS:
        raise SystemExit("The current clean export must include all three sources.")

    destination = clean if args.publish else root / "data/intermediate/expanded_export"
    if destination != clean and destination.exists():
        raise SystemExit(f"Preview already exists: {destination}. Move it aside before rebuilding.")
    work = Path(mkdtemp(prefix="frame_expansion_", dir=root / "data/intermediate"))
    output = work / "export"
    output.mkdir()
    prefix = (destination / "images").relative_to(root).as_posix()
    try:
        build(root, output, prefix, fields, current, args.lung_range_174)
        if args.publish:
            backup = work / "previous_clean"
            clean.rename(backup)
            try:
                output.rename(clean)
            except BaseException:
                backup.rename(clean)
                raise
            print(f"Previous clean export archived at {backup.relative_to(root)}")
        else:
            output.rename(destination)
            work.rmdir()
    except Exception:
        print(f"Partial build retained at {work.relative_to(root)}")
        raise
    print(f"Expanded export: {destination.relative_to(root)}")


if __name__ == "__main__":
    main()

