import csv
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageChops, ImageOps


ROOT = Path(__file__).resolve().parents[1]
KATUMBA = ROOT / "data/intermediate/katumba/staged/staged_manifest.csv"
CLEAN = ROOT / "data/clean/manifests/normalization_manifest.csv"
REPORT = ROOT / "data/intermediate/katumba/overlap_audit.csv"
MAX_HASH_DISTANCE = 8

REPORT_FIELDS = [
    "audit_status",
    "comparison",
    "hash_distance",
    "katumba_group_id",
    "katumba_class",
    "katumba_staged_path",
    "katumba_source_path",
    "other_dataset",
    "other_group_id",
    "other_class",
    "other_image_path",
    "other_source_path",
    "review_decision",
    "review_notes",
]


def load_images(manifest, path_column, exclude_katumba=False):
    records = []

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "group_id", "class", "source_path", path_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{manifest.name} is missing columns: {sorted(missing)}")

        for row in reader:
            if exclude_katumba and row["dataset"].strip().casefold() == "katumba":
                continue

            relative_path = row[path_column]
            if not relative_path or not row["group_id"]:
                raise ValueError(f"Missing image path or group ID in {manifest.name}")

            try:
                with Image.open(ROOT / relative_path) as source:
                    image = ImageOps.exif_transpose(source).convert("L")
            except OSError as error:
                raise RuntimeError(f"Could not read {relative_path}: {error}") from error

            exact_hash = sha256(
                f"{image.width}x{image.height}:".encode() + image.tobytes()
            ).hexdigest()

            # Ignore near-black padding for the perceptual comparison.
            foreground = image.point(lambda pixel: 255 if pixel > 12 else 0)
            bounds = foreground.getbbox()
            if bounds is None or min(image.size) < 2:
                raise ValueError(f"No usable foreground in {relative_path}")

            scan = image.crop(bounds)
            if not ImageChops.difference(
                scan, Image.new("L", scan.size, scan.getpixel((0, 0)))
            ).getbbox():
                raise ValueError(f"Uniform foreground in {relative_path}")

            small = scan.resize((9, 8), Image.Resampling.LANCZOS)
            pixels = small.tobytes()
            perceptual_hash = 0
            for y in range(8):
                for x in range(8):
                    offset = y * 9 + x
                    perceptual_hash = (
                        (perceptual_hash << 1)
                        | (pixels[offset] > pixels[offset + 1])
                    )

            records.append({
                **row,
                "audit_image_path": relative_path,
                "exact_hash": exact_hash,
                "perceptual_hash": perceptual_hash,
            })

    return records


def compare(candidate, other, comparison):
    distance = (
        candidate["perceptual_hash"] ^ other["perceptual_hash"]
    ).bit_count()

    if candidate["exact_hash"] == other["exact_hash"]:
        status = "exact_match"
    elif distance <= MAX_HASH_DISTANCE:
        status = "possible_match"
    else:
        return None

    return {
        "audit_status": status,
        "comparison": comparison,
        "hash_distance": distance,
        "katumba_group_id": candidate["group_id"],
        "katumba_class": candidate["class"],
        "katumba_staged_path": candidate["audit_image_path"],
        "katumba_source_path": candidate["source_path"],
        "other_dataset": other["dataset"],
        "other_group_id": other["group_id"],
        "other_class": other["class"],
        "other_image_path": other["audit_image_path"],
        "other_source_path": other["source_path"],
    }


def main():
    staged = load_images(KATUMBA, "staged_path")
    clean = load_images(CLEAN, "normalized_path", exclude_katumba=True)
    if not staged or not clean:
        raise SystemExit("The audit needs staged Katumba images and existing clean images.")

    matches = []
    flagged_groups = set()

    for index, candidate in enumerate(staged):
        comparisons = [
            (other, "existing_clean")
            for other in clean
        ]
        comparisons.extend(
            (other, "different_katumba_group")
            for other in staged[index + 1:]
            if other["group_id"] != candidate["group_id"]
        )

        for other, comparison in comparisons:
            result = compare(candidate, other, comparison)
            if result is not None:
                matches.append(result)
                flagged_groups.add(candidate["group_id"])
                if comparison == "different_katumba_group":
                    flagged_groups.add(other["group_id"])

    matches.sort(key=lambda row: (
        row["audit_status"] != "exact_match",
        row["hash_distance"],
        row["katumba_group_id"],
        row["katumba_staged_path"],
        row["other_image_path"],
    ))

    # Preserve previous decisions when the same pair appears on a rerun.
    previous = {}
    if REPORT.exists():
        with REPORT.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = (row["katumba_staged_path"], row["other_image_path"])
                previous[key] = (row["review_decision"], row["review_notes"])

    temporary = REPORT.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in matches:
            key = (row["katumba_staged_path"], row["other_image_path"])
            row["review_decision"], row["review_notes"] = previous.get(key, ("", ""))
            writer.writerow(row)
    temporary.replace(REPORT)

    exact_count = sum(row["audit_status"] == "exact_match" for row in matches)
    group_count = len({row["group_id"] for row in staged})
    print(f"Compared {len(staged)} staged stills with {len(clean)} clean stills.")
    print(f"Exact image pairs: {exact_count}")
    print(f"Possible image pairs: {len(matches) - exact_count}")
    print(f"Katumba groups with flagged pairs: {len(flagged_groups)} of {group_count}")
    print(f"Audit report: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()