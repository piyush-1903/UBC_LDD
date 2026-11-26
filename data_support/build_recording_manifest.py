from collections import defaultdict
from pathlib import Path

import pandas as pd

repo_dir = Path(__file__).resolve().parents[1]
dataset_dir = repo_dir / "data" / "raw" / "born_pocus"
intermediate_dir = repo_dir / "data" / "intermediate"
intermediate_dir.mkdir(parents=True, exist_ok=True)
output_path = intermediate_dir / "recordings.csv"

metadata = pd.read_csv(
    dataset_dir / "dataset_metadata.csv",
    encoding="cp1252",
    dtype=str,
    keep_default_na=False,
)
hosted = metadata.loc[
    metadata["InThisRepo"].str.strip().str.casefold().eq("yes")
].copy()

label_map = {
    "regular": "healthy",
    "bacterial pneumonia": "pneumonia",
    "viral pneumonia": "pneumonia",
    "covid-19": "covid",
}
hosted["class"] = hosted["Label"].str.strip().str.casefold().map(label_map)

files_by_key = defaultdict(list)
for folder in ("pocus_images", "pocus_videos"):
    for path in (dataset_dir / folder).rglob("*"):
        if path.is_file():
            relative = path.relative_to(dataset_dir)
            key = (relative.parent.as_posix().casefold(), path.stem.casefold())
            files_by_key[key].append(relative.as_posix())

records = []
for _, row in hosted.iterrows():
    folder = row["Current location"].strip().replace("\\", "/")
    folder = folder.removeprefix("data/").rstrip("/")
    filename = row["Filename"].strip()
    matches = files_by_key[(folder.casefold(), Path(filename).stem.casefold())]

    if len(matches) == 1:
        status = "matched"
    elif matches:
        status = "ambiguous"
    else:
        status = "missing"

    records.append({
        "source_location": row["Current location"],
        "source_filename": filename,
        "source_url": row["URL (Video Name)"],
        "source_id": row["Source ID"],
        "source_patient_id": row["Patient ID / Name"].strip(),
        "original_label": row["Label"],
        "class": row["class"],
        "probe": row["Probe"],
        "license": row["License"],
        "relative_path": matches[0] if status == "matched" else "",
        "file_status": status,
    })

manifest = pd.DataFrame(records)
manifest.to_csv(output_path, index=False)
print(pd.crosstab(manifest["class"], manifest["file_status"]).to_string())
print(f"\nWrote {len(manifest)} rows to {output_path}")

matched = manifest.loc[manifest["file_status"].eq("matched")].copy()
id_columns = ["source_id", "source_patient_id"]

for column in id_columns:
    matched[column] = matched[column].fillna("").str.strip()

placeholders = {"", "na", "n/a", "nd", "n.d.", "unknown", "none", "-"}
has_patient_key = ~matched[id_columns].apply(
    lambda column: column.str.casefold().isin(placeholders)
).any(axis=1)

groups = matched.loc[has_patient_key].groupby(id_columns).agg(
    label=("class", "first"),
    label_count=("class", "nunique"),
)
consistent = groups.loc[groups["label_count"].eq(1)]

print("\nProvisional patient groups by class:")
print(consistent["label"].value_counts().to_string())
print("\nMatched recordings missing a usable source or patient ID:", (~has_patient_key).sum())
print("Patient groups with conflicting labels:", groups["label_count"].gt(1).sum())

conflicting_keys = groups.loc[groups["label_count"].gt(1)].reset_index()[id_columns]
conflicting_rows = matched.merge(conflicting_keys, on=id_columns)

print("\nRecordings with conflicting labels:")
print(
    conflicting_rows[
        ["source_id", "original_label", "source_filename", "source_url"]
    ].to_string(index=False)
)

candidates = matched.merge(
    consistent.reset_index()[id_columns],
    on=id_columns,
    how="inner",
    validate="many_to_one",
)

candidates.to_csv(intermediate_dir / "baseline_candidates.csv", index=False)

print("\nCandidate recordings and patient groups:")
print(candidates.groupby("class").size().rename("recordings"))
print(candidates.drop_duplicates(id_columns).groupby("class").size().rename("groups"))