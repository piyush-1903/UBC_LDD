import argparse
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd


LABELS = {"pneumonia": "pneumonia", "covid": "covid", "normal": "healthy"}


def url_keys(text):
    keys = set()
    for url in re.findall(r"https?://[^\s,;)]+", text):
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        path = parsed.path.rstrip("/")
        if host and path:
            keys.add((host, path, parsed.query))
    return keys


def is_clip_url(key):
    host, path, _ = key
    return Path(path).suffix.lower() in {".mp4", ".gif", ".mov", ".webm", ".avi"} or (
        host in {"vimeo.com", "player.vimeo.com"} and path.split("/")[-1].isdigit()
    )


def read_csv(path, **kwargs):
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data = args.repo_root / "data"
    intermediate = data / "intermediate"
    born_path = intermediate / "born_pocus" / "recordings.csv"
    if not born_path.is_file():
        born_path = intermediate / "recordings.csv"
    born = read_csv(born_path)
    covidx = read_csv(data / "raw/covidx_us/video_metadata.csv", encoding="cp1252")
    output = args.output_dir or intermediate / "covidx_us"

    by_url = defaultdict(set)
    for index, row in born.loc[born["file_status"].eq("matched")].iterrows():
        for key in url_keys(row["source_url"]):
            by_url[key].add(index)

    audit = []
    for row in covidx.to_dict("records"):
        original_label = row["class"].strip().casefold()
        if original_label not in LABELS:
            continue

        keys = url_keys(row["url"])
        direct = {index for key in keys if is_clip_url(key) for index in by_url.get(key, ())}
        shared_page = {index for key in keys | url_keys(row["link"])
                       if not is_clip_url(key) for index in by_url.get(key, ())}
        matches = born.loc[sorted(direct or shared_page)]
        labels = sorted(set(matches["class"]))
        if direct:
            status = "same_url" if labels == [LABELS[original_label]] else "same_url_label_conflict"
        elif shared_page:
            status = "shared_page_review"
        else:
            status = "needs_case_review"

        audit.append({
            "covidx_id": row["id"],
            "source": row["source"],
            "upstream_url": row["url"],
            "case_url": row["link"],
            "patient": row["patient"].strip(),
            "case_no": row["case_no"],
            "original_label": row["class"],
            "candidate_label": LABELS[original_label],
            "website_label": row["class_on_website"],
            "license": row["license"],
            "born_recordings": " | ".join(matches["source_filename"]),
            "born_labels": " | ".join(labels),
            "audit_status": status,
        })

    report = pd.DataFrame(audit)
    direct_statuses = {"same_url", "same_url_label_conflict"}
    linked = report.loc[report["patient"].ne("") & report["audit_status"].isin(direct_statuses)]
    linked_keys = set(zip(linked["source"], linked["patient"]))
    report["patient_group_has_direct_match"] = [
        (source, patient) in linked_keys for source, patient in zip(report["source"], report["patient"])
    ]
    output.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        subset = report.loc[report["original_label"].str.strip().str.casefold().eq(label)]
        path = output / f"{label}_overlap_audit.csv"
        subset.to_csv(path, index=False)
        print(f"{label}: {len(subset)} entries | {subset['audit_status'].value_counts().to_dict()}")
        unresolved = subset.loc[~subset["audit_status"].isin(direct_statuses)]
        tagged = unresolved.loc[unresolved["patient"].ne("") & ~unresolved["patient_group_has_direct_match"]]
        print(f"  Unlinked ID-backed groups: {len(tagged.drop_duplicates(['source', 'patient']))}; "
              f"unresolved clips without IDs: {unresolved['patient'].eq('').sum()}")
        print(f"  {path}")


if __name__ == "__main__":
    main()

