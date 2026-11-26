import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KATUMBA = ROOT / "data/intermediate/katumba"
AUDIT = KATUMBA / "overlap_audit.csv"
RESOLVED = KATUMBA / "staged/resolved_staged_manifest.csv"
OUTPUT = KATUMBA / "overlap_review.csv"


def main():
    with RESOLVED.open(newline="", encoding="utf-8-sig") as handle:
        retained = {row["staged_path"] for row in csv.DictReader(handle)}

    if not retained:
        raise SystemExit("The resolved manifest contains no retained stills.")

    reviewed = []
    counts = Counter()
    excluded_pairs = 0

    with AUDIT.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = [
            *reader.fieldnames,
            "overlap_disposition",
            "disposition_reason",
        ]

        for row in reader:
            comparison = row["comparison"]
            status = row["audit_status"]
            if comparison not in {"existing_clean", "different_katumba_group"}:
                raise ValueError(f"Unexpected comparison: {comparison!r}")
            if status not in {"exact_match", "possible_match"}:
                raise ValueError(f"Unexpected audit status: {status!r}")

            within_katumba = comparison == "different_katumba_group"
            if row["katumba_staged_path"] not in retained or (
                within_katumba and row["other_image_path"] not in retained
            ):
                excluded_pairs += 1
                continue

            if status == "exact_match":
                disposition = "review_exact_match"
                reason = "Both images remain in scope and have identical pixels."
            elif within_katumba:
                disposition = "review_within_katumba"
                reason = "Similar images are assigned to different Katumba patient groups."
            else:
                disposition = "auto_clear_cross_source"
                reason = "Perceptual similarity only, with separate documented cohort provenance."

            row["overlap_disposition"] = disposition
            row["disposition_reason"] = reason
            reviewed.append(row)
            counts[(comparison, row["other_dataset"], disposition)] += 1

    reviewed.sort(key=lambda row: (
        row["overlap_disposition"] != "review_exact_match",
        row["overlap_disposition"] == "auto_clear_cross_source",
        int(row["hash_distance"]),
    ))

    temporary = OUTPUT.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reviewed)
    temporary.replace(OUTPUT)

    print(f"Retained Katumba stills: {len(retained)}")
    print(f"Pairs involving excluded stills: {excluded_pairs}")
    for (comparison, dataset, disposition), count in sorted(counts.items()):
        print(f"{comparison} | {dataset} | {disposition}: {count}")

    pending = sum(
        count for (_, _, disposition), count in counts.items()
        if disposition.startswith("review_")
    )
    print(f"Pairs requiring review: {pending}")
    print(f"Report: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()