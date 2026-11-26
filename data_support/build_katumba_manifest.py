import argparse
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LABELS = {"healthy lung": "healthy", "probably covid": "covid",
          "diseased lung but probably not covid": "other"}


def normalized_key(value):
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def workbook_rows(path):
    with ZipFile(path) as book:
        strings = [
            "".join(node.text or "" for node in item.findall(".//s:t", NS))
            for item in ET.fromstring(book.read("xl/sharedStrings.xml"))
        ] if "xl/sharedStrings.xml" in book.namelist() else []
        sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
    for row in sheet.findall(".//s:row", NS):
        values = {}
        for cell in row.findall("s:c", NS):
            column = re.sub(r"\d", "", cell.attrib["r"])
            value = cell.findtext("s:v", "", NS)
            if cell.get("t") == "s" and value:
                value = strings[int(value)]
            elif cell.get("t") == "inlineStr":
                value = "".join(t.text or "" for t in cell.findall(".//s:t", NS))
            values[column] = value.strip()
        yield values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    raw = root / "data/raw/katumba"
    dataset = raw / "dataset"
    rows = workbook_rows(raw / "metadata.xlsx")
    columns = {name: key for key, name in next(rows).items()}
    patient_col = columns["Patient Number"]
    label_col = columns["Conclusion"]
    matches_by_name = defaultdict(set)
    for row in rows:
        patient = normalized_key(row.get(patient_col, ""))
        label = row.get(label_col, "")
        for value in row.values():
            path = Path(value)
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                matches_by_name[normalized_key(path.stem)].add((patient, label))

    manifest = []
    labels_by_patient = defaultdict(set)
    for path in sorted(dataset.glob("*/*/*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        source_split, folder_label, _ = path.relative_to(dataset).parts
        matches = matches_by_name[normalized_key(path.stem)]
        patient, original_label = next(iter(matches)) if len(matches) == 1 else ("", "")
        label = LABELS.get(original_label.casefold(), "")
        if not matches:
            status = "no_metadata_match"
        elif len(matches) != 1 or not patient:
            status = "unresolved_patient"
        elif label != folder_label:
            status = "label_disagreement"
        elif label not in {"covid", "healthy"}:
            status = "outside_target_classes"
        else:
            status = "candidate"
        group_id = "katumba_" + hashlib.sha256(patient.encode()).hexdigest() if patient else ""
        if group_id and label:
            labels_by_patient[group_id].add(label)
        manifest.append({
            "dataset": "katumba",
            "source_path": path.relative_to(root).as_posix(),
            "source_split": source_split,
            "folder_label": folder_label,
            "original_label": original_label,
            "class": label,
            "group_id": group_id,
            "status": status,
        })

    if not manifest:
        raise SystemExit(f"No images found in {dataset}")
    for row in manifest:
        if row["status"] == "candidate" and len(labels_by_patient[row["group_id"]]) > 1:
            row["status"] = "patient_label_conflict"

    output = root / "data/intermediate/katumba/intake_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest[0])
        writer.writeheader()
        writer.writerows(manifest)
    for label in ("healthy", "covid"):
        accepted = [r for r in manifest if r["status"] == "candidate" and r["class"] == label]
        print(f"{label}: {len(accepted)} images, {len({r['group_id'] for r in accepted})} groups")
    print(f"Wrote {len(manifest)} rows to {output.relative_to(root)}")


if __name__ == "__main__":
    main()

