import csv
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / "data/intermediate/katumba/intake_manifest.csv"
STAGING = ROOT / "data/intermediate/katumba/staged"
CANVAS_SIZE = (512, 512)
MAX_STILLS = 3


def normalize(source_path):
    with Image.open(source_path) as source:
        grayscale = ImageOps.exif_transpose(source).convert("L")
        return ImageOps.pad(
            grayscale,
            CANVAS_SIZE,
            method=Image.Resampling.LANCZOS,
            color=0,
        )


def main():
    groups = defaultdict(list)
    with INTAKE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        input_fields = list(reader.fieldnames or [])
        for row in reader:
            if row["status"] == "candidate":
                groups[row["group_id"]].append(row)

    if not groups:
        raise SystemExit("No candidate patient groups found in the intake manifest.")

    selected = []
    duplicate_count = 0

    for group_id, rows in sorted(groups.items()):
        if not group_id or len({row["class"] for row in rows}) != 1:
            raise ValueError(f"Missing group ID or conflicting labels in {group_id!r}")

        seen_pixels = set()
        unique = []
        for row in sorted(rows, key=lambda item: Path(item["source_path"]).name.casefold()):
            image = normalize(ROOT / row["source_path"])
            digest = sha256(image.tobytes()).hexdigest()
            if digest in seen_pixels:
                duplicate_count += 1
                continue
            seen_pixels.add(digest)
            unique.append((row, digest))

        count = min(MAX_STILLS, len(unique))
        # Spread selections across the filename order for a reproducible first pass.
        indices = [
            round(i * (len(unique) - 1) / (count - 1)) if count > 1 else 0
            for i in range(count)
        ]
        default_index = indices[len(indices) // 2]

        for number, index in enumerate(indices, start=1):
            row, digest = unique[index]
            destination = STAGING / "images" / f"{group_id}_{number:02d}_{digest[:12]}.png"
            selected.append((row, number, index == default_index, destination))

    (STAGING / "images").mkdir(parents=True, exist_ok=True)
    manifest_path = STAGING / "staged_manifest.csv"
    temporary_path = manifest_path.with_suffix(".tmp")
    fields = input_fields + ["still_number", "is_default", "staged_path"]
    still_counts = Counter()
    group_counts = defaultdict(set)

    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, number, is_default, destination in selected:
            normalize(ROOT / row["source_path"]).save(destination)
            writer.writerow({
                **row,
                "still_number": number,
                "is_default": is_default,
                "staged_path": destination.relative_to(ROOT).as_posix(),
            })
            still_counts[row["class"]] += 1
            group_counts[row["class"]].add(row["group_id"])

    temporary_path.replace(manifest_path)

    for label in sorted(still_counts):
        print(f"{label}: {still_counts[label]} stills, {len(group_counts[label])} groups")
    print(f"Skipped {duplicate_count} exact duplicates within patient groups.")
    print(f"Staged {len(selected)} stills across {len(groups)} groups.")
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()