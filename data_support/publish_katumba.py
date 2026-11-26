import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE = ROOT / "data/intermediate/katumba"
STAGED = INTERMEDIATE / "staged/resolved_staged_manifest.csv"
REVIEW = INTERMEDIATE / "overlap_review.csv"
CLEAN = ROOT / "data/clean"
MANIFEST = CLEAN / "manifests/normalization_manifest.csv"
CANVAS = (512, 512)
HELD_IMAGES = INTERMEDIATE / "label_hold" / "images"

# Reviewed workbook symptom histories explicitly report a positive COVID RDT.
COVID_RDT_EVIDENCE = {
    "katumba_14eec5b52594640d1f6ddfdb11424c541b517d2ae6a5cc5dba1b9079305aed54": "timing unspecified",
    "katumba_160eb79fe88ac6b1551f352052ba1dbac0bba22039d8d0bbf4d88c3d81e638d3": "two weeks before the recorded history",
    "katumba_4d3bd5f0f61eea90c14dd8014f738aaefc50698f67ca23e22db3ed491028fccb": "two weeks before the recorded history",
    "katumba_efcc56dcdbfbb57e2297ac4a170b26830f2f54e8c18df989b5dd8b2c096fbdde": "timing unspecified",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path, fields, rows):
    with NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8",
        dir=path.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise

    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def group_counts(rows):
    groups = {(row["class"], row["group_id"]) for row in rows}
    return Counter(label for label, _ in groups)


def main():
    _, staged = read_csv(STAGED)
    review_fields, review = read_csv(REVIEW)
    clean_fields, clean = read_csv(MANIFEST)

    still_counts = Counter(row["class"] for row in staged)
    if still_counts != {"healthy": 117, "covid": 70} or group_counts(staged) != {
        "healthy": 40, "covid": 24,
    }:
        raise SystemExit("The staged counts differ from the reviewed Katumba batch.")

    retained = {row["staged_path"]: row for row in staged}
    if len(retained) != len(staged):
        raise SystemExit("Duplicate staged paths in the resolved manifest.")

    groups = defaultdict(list)
    for row in staged:
        if row["dataset"] != "katumba" or not row["group_id"].startswith("katumba_"):
            raise SystemExit("Unexpected dataset or group ID in the resolved manifest.")
        groups[row["group_id"]].append(row)
    for group_id, rows in groups.items():
        defaults = sum(row["is_default"].strip().casefold() == "true" for row in rows)
        if len(rows) > 3 or defaults != 1 or len({row["class"] for row in rows}) != 1:
            raise SystemExit(f"Invalid staged patient group: {group_id}")

    expected_review = {
        ("review_within_katumba", "katumba"): 44,
        ("auto_clear_cross_source", "born_pocus"): 48,
        ("auto_clear_cross_source", "covidx_us"): 1,
    }
    actual_review = Counter(
        (row["overlap_disposition"], row["other_dataset"]) for row in review
    )
    pair_keys = {
        (row["katumba_staged_path"], row["other_image_path"]) for row in review
    }
    if actual_review != expected_review or len(pair_keys) != len(review):
        raise SystemExit("The overlap report differs from the reviewed batch.")

    existing = [row for row in clean if row["dataset"].strip().casefold() != "katumba"]
    if not existing:
        raise SystemExit("The shared manifest contains no existing source rows.")
    existing_paths = {
        row["normalized_path"]: row for row in existing
    }

    for row in review:
        left = retained.get(row["katumba_staged_path"])
        within = row["overlap_disposition"] == "review_within_katumba"
        right_paths = retained if within else existing_paths
        right = right_paths.get(row["other_image_path"])
        expected_comparison = "different_katumba_group" if within else "existing_clean"
        if (
            row["audit_status"] != "possible_match"
            or row["comparison"] != expected_comparison
            or left is None
            or right is None
            or left["group_id"] != row["katumba_group_id"]
            or right["group_id"] != row["other_group_id"]
            or right["dataset"] != row["other_dataset"]
        ):
            raise SystemExit("An audit pair differs from the current manifests.")
        if row["review_decision"].strip().casefold() not in {"", "distinct"}:
            raise SystemExit("The report contains a conflicting review decision.")
        if within:
            row["review_decision"] = "distinct"
            if not row["review_notes"].strip():
                row["review_notes"] = (
                    "Side-by-side review found different scan features. "
                    "Cleared for image overlap."
                )

    required_fields = {"dataset", "group_id", "class", "normalized_path"}
    if not required_fields.issubset(clean_fields):
        raise SystemExit("The shared normalization manifest is missing required columns.")

    fields = list(dict.fromkeys([*clean_fields, "original_label", "source_split"]))
    covid_groups = {row["group_id"] for row in staged if row["class"] == "covid"}
    if not COVID_RDT_EVIDENCE.keys() <= covid_groups:
        raise SystemExit("A reviewed RDT-positive group is absent from the staged COVID rows.")
    selected = [
        row for row in staged
        if row["class"] != "covid" or row["group_id"] in COVID_RDT_EVIDENCE
    ]
    held = [
        row for row in staged
        if row["class"] == "covid" and row["group_id"] not in COVID_RDT_EVIDENCE
    ]
    if len(held) != 59:
        raise SystemExit("Unexpected number of Katumba COVID stills on hold.")

    publications = []
    for row in selected:
        staged_image = ROOT / row["staged_path"]
        raw_image = ROOT / row["source_path"]
        with Image.open(staged_image) as image:
            image.load()
            if image.format != "PNG" or image.mode != "L" or image.size != CANVAS:
                raise SystemExit(f"Expected a 512 × 512 grayscale PNG: {staged_image}")
        with Image.open(raw_image) as image:
            width, height = ImageOps.exif_transpose(image).size

        destination = CLEAN / "images" / staged_image.name
        relative_path = destination.relative_to(ROOT).as_posix()
        if relative_path in existing_paths:
            raise SystemExit(f"Image path belongs to another source: {relative_path}")
        if destination.exists() and destination.read_bytes() != staged_image.read_bytes():
            raise SystemExit(f"Image already exists with different contents: {destination}")

        record = {
            **row,
            "source_path": raw_image.relative_to(ROOT / "data/raw/katumba").as_posix(),
            "frame_index": "",
            "origin": "original_still",
            "additional_frame_from_same_clip": "False",
            "image_path": row["source_path"],
            "status": "approved",
            "source_id": raw_image.stem,
            "original_width": width,
            "original_height": height,
            "crop_x0": 0,
            "crop_y0": 0,
            "crop_x1": width,
            "crop_y1": height,
            "output_width": CANVAS[0],
            "output_height": CANVAS[1],
            "normalized_path": relative_path,
            "normalization_qc": "needs_crop_review",
        }
        publications.append((
            staged_image, destination, {field: record.get(field, "") for field in fields}
        ))

    # Validate files before moving anything out of the shared image directory.
    held_files = []
    for row in held:
        staged_image = ROOT / row["staged_path"]
        clean_image = CLEAN / "images" / staged_image.name
        relative_path = clean_image.relative_to(ROOT).as_posix()
        if relative_path in existing_paths:
            raise SystemExit(f"Held image path belongs to another dataset: {relative_path}")
        if not clean_image.exists():
            continue
        image_bytes = clean_image.read_bytes()
        if image_bytes != staged_image.read_bytes():
            raise SystemExit(f"Held clean image differs from its staged copy: {clean_image}")
        archived = HELD_IMAGES / clean_image.name
        if archived.exists() and archived.read_bytes() != image_bytes:
            raise SystemExit(f"Held image archive already contains different bytes: {archived}")
        held_files.append((clean_image, archived))

    (CLEAN / "images").mkdir(exist_ok=True)
    for source, destination, _ in publications:
        if destination.exists():
            continue
        with NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    combined = existing + [record for _, _, record in publications]
    write_csv(REVIEW, review_fields, review)
    write_csv(MANIFEST, fields, combined)

    # Update the manifest first so a failed move can be retried without broken rows.
    if held_files:
        HELD_IMAGES.mkdir(parents=True, exist_ok=True)
    for clean_image, archived in held_files:
        os.replace(clean_image, archived)

    print(f"Held {len(held)} Katumba COVID stills across {len(covid_groups) - len(COVID_RDT_EVIDENCE)} groups.")
    print(f"Archived {len(held_files)} previously published images under {HELD_IMAGES.relative_to(ROOT)}.")
    print(f"Published {len(publications)} Katumba stills across {sum(group_counts(selected).values())} groups.")
    print(f"Combined dataset: {len(combined)} stills across {sum(group_counts(combined).values())} groups.")
    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()