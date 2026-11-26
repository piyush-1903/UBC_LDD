import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KATUMBA = ROOT / "data/intermediate/katumba"
STAGED = KATUMBA / "staged/staged_manifest.csv"
AUDIT = KATUMBA / "overlap_audit.csv"
RESOLVED = KATUMBA / "staged/resolved_staged_manifest.csv"

KEEP_GROUP = "katumba_160eb79fe88ac6b1551f352052ba1dbac0bba22039d8d0bbf4d88c3d81e638d3"
DROP_GROUP = "katumba_25b076033cd9f4a5cb9fd3b68a23197386ef613197e71a8f94f761d762f7ef14"


def main():
    with STAGED.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)

    keep_rows = [row for row in rows if row["group_id"] == KEEP_GROUP]
    drop_rows = [row for row in rows if row["group_id"] == DROP_GROUP]

    for group_rows, split in ((keep_rows, "train"), (drop_rows, "validation")):
        if len(group_rows) != 3 or any(
            row["class"] != "covid" or row["source_split"] != split
            for row in group_rows
        ):
            raise SystemExit(f"Expected three COVID stills from the {split} group.")

    matched_pairs = set()
    with AUDIT.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (
                row["audit_status"] != "exact_match"
                or row["comparison"] != "different_katumba_group"
            ):
                continue

            left = row["katumba_group_id"]
            right = row["other_group_id"]
            if (left, right) == (KEEP_GROUP, DROP_GROUP):
                matched_pairs.add(
                    (row["katumba_staged_path"], row["other_image_path"])
                )
            elif (left, right) == (DROP_GROUP, KEEP_GROUP):
                matched_pairs.add(
                    (row["other_image_path"], row["katumba_staged_path"])
                )

    keep_paths = {row["staged_path"] for row in keep_rows}
    drop_paths = {row["staged_path"] for row in drop_rows}
    if (
        len(matched_pairs) != 3
        or len(keep_paths) != 3
        or len(drop_paths) != 3
        or {left for left, _ in matched_pairs} != keep_paths
        or {right for _, right in matched_pairs} != drop_paths
    ):
        raise SystemExit("The audit must match all three stills one-to-one.")

    resolved_rows = [row for row in rows if row["group_id"] != DROP_GROUP]
    temporary = RESOLVED.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(resolved_rows)
    temporary.replace(RESOLVED)

    still_counts = Counter(row["class"] for row in resolved_rows)
    groups = {(row["class"], row["group_id"]) for row in resolved_rows}
    group_counts = Counter(label for label, _ in groups)

    print(f"Excluded {len(drop_rows)} duplicate stills from one patient group.")
    for label in sorted(still_counts):
        print(f"{label}: {still_counts[label]} stills, {group_counts[label]} groups")
    print(f"Total: {len(resolved_rows)} stills, {len(groups)} groups")
    print(f"Manifest: {RESOLVED.relative_to(ROOT)}")


if __name__ == "__main__":
    main()