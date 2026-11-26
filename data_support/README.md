# Dataset Preprocessing

These scripts prepare lung ultrasound stills from Born POCUS, COVIDx-US, and Katumba for the capstone’s three-class classifier. The shared dataset uses `healthy`, `pneumonia`, and `covid` labels.

| Script | What it does | Why we use it |
|---|---|---|
| [`build_recording_manifest.py`](build_recording_manifest.py) | Matches Born POCUS metadata to local media and writes the recording and candidate CSVs. | Establishes which recordings and provisional patient groups can enter extraction. |
| [`extract_baseline_stills.py`](extract_baseline_stills.py) | Selects up to three stills per Born POCUS patient group and writes the multi-view manifest. | Retains different views while supporting a reproducible one-still baseline. |
| [`normalize_stills.py`](normalize_stills.py) | Converts Born POCUS stills to 512 × 512 grayscale PNGs and creates the shared clean manifest. | Establishes the shared image format and loading path. |
| [`audit_covidx_overlap.py`](audit_covidx_overlap.py) | Compares COVIDx-US metadata with the Born POCUS recording inventory and writes one audit CSV per target class. | Flags shared provenance and label conflicts before new clips enter the pipeline. |
| [`extract_covidx_stills.py`](extract_covidx_stills.py) | Samples draft frames from the initial COVIDx-US pneumonia review clips. | Supports visual and duplicate review of candidate recordings. |
| [`integrate_covidx_shortlist.py`](integrate_covidx_shortlist.py) | Stages normalized frames from shortlisted COVIDx-US clips. With `--publish`, adds the reviewed selections to the shared export. | Keeps the review batch separate from the selected dataset and preserves patient grouping. |
| [`mask_covidx_overlays.py`](mask_covidx_overlays.py) | Generates masking previews and, with `--apply`, updates the staged manifest to point to masked images. | Reduces visible scanner and source cues that a model might learn. |
| [`build_katumba_manifest.py`](build_katumba_manifest.py) | Matches processed images across the source splits to the workbook and writes the intake manifest. | Maps labels and assigns one stable group ID per source patient. |
| [`stage_katumba_stills.py`](stage_katumba_stills.py) | Selects up to three distinct stills per group and stages 512 × 512 grayscale PNGs. | Prepares a reproducible review batch with one default still per group. |
| [`audit_katumba_overlap.py`](audit_katumba_overlap.py) | Checks staged stills against existing clean images and across Katumba groups using exact pixels and perceptual hashes. | Flags possible reused images and cross-group duplicates. |
| [`resolve_katumba_duplicate.py`](resolve_katumba_duplicate.py) | Excludes the three exact duplicate copies from one Katumba COVID group in a resolved manifest. | Keeps the duplicated case under a single provisional group. |
| [`review_katumba_overlap.py`](review_katumba_overlap.py) | Filters out pairs involving excluded stills and records the disposition of each remaining pair. | Clears cross-source perceptual similarities and identifies within-Katumba pairs for visual review. |
| [`publish_katumba.py`](publish_katumba.py) | Records the reviewed decisions and publishes the retained stills and metadata. | Adds the reviewed Katumba batch to the shared export. |

## Born POCUS pipeline

### `build_recording_manifest.py`

Reads the Born POCUS metadata under `data/raw/born_pocus/` and matches it to the local image and video files. It maps the original labels to the project classes and uses the source and patient identifiers to create provisional groups. It records unmatched and ambiguous files for review, and excludes groups with conflicting labels from the candidate list.

Outputs `recordings.csv` and `baseline_candidates.csv` under `data/intermediate/born_pocus/`. The recording inventory provides the provenance used by the COVIDx-US overlap audit.

### `extract_baseline_stills.py`

Reads the candidate list and selects up to three stills per group. It prefers existing stills and distinct source clips, then samples additional frames if needed. It skips exact pixel duplicates within a group and designates one still as the default. The multi-view manifest records the source path and frame index so a selected still can be traced back to its recording.

Writes the stills and `multi_view_manifest.csv` under `data/intermediate/born_pocus/`.

### `normalize_stills.py`

Reads the Born POCUS multi-view manifest and applies configured crops, converts the stills to 8-bit grayscale, and rescales them proportionally. It upscales or downscales as needed, then centers each image on a black 512 × 512 canvas. The fixed canvas gives model inputs a consistent shape. Native resolution, field of view, and physical scan depth can vary by source.

The default run builds the Born POCUS clean export. Use `--pilot` to inspect a sample of before-and-after images. Use the default run as the base when rebuilding the published combined dataset. The optional `--include-covidx-drafts` mode serves draft review.

## COVIDx-US pipeline

### `audit_covidx_overlap.py`

Reads the COVIDx-US video metadata and compares upstream URLs against the matched Born POCUS recording inventory across all labels. It distinguishes direct clip URL matches from shared case-page links and flags possible label conflicts. It also identifies COVIDx-US patient groups linked to a direct match.

Writes `pneumonia_overlap_audit.csv`, `covid_overlap_audit.csv`, and `normal_overlap_audit.csv` under `data/intermediate/covidx_us/`. Follow up with source case descriptions and clip comparisons where needed. Different URLs can still identify the same recording or patient. A normal-looking scan also needs case evidence before it is assigned the `healthy` label.

### `extract_covidx_stills.py`

Samples frames at approximately 25%, 50%, and 75% of each configured clip in the initial pneumonia review batch. It skips exact duplicate frames within a clip and records each sampled frame in `draft_stills_manifest.csv`. The PNGs and CSV stay under `data/intermediate/covidx_us/`.

Use this script when revisiting the initial candidate review. The approved shortlist has its own staging and export workflow below.

### `integrate_covidx_shortlist.py`

Reads the shortlisted videos from `data/raw/covidx_us/videos/` and stages sample frames under `data/intermediate/covidx_us/shortlist/`. It applies a clip-specific crop where needed, including the crop that isolates a lung panel from the combined heart and lung views in clip `174`. It uses the same 512 × 512 normalization as Born POCUS. The `frames.csv` inventory records the class, patient group, source clip, frame index, and crop bounds. Two clips from one case can share a group, as with Kalafat clips `199` and `200`.

After image and mask review, `--publish` reads the selected paths from `approved.csv`, enforces a maximum of three stills per group, and updates the COVIDx-US portion of the shared clean manifest. It expects the approved CSV to contain the full COVIDx-US selection intended for export. Keep optional source reuse references in its `reuse_evidence` column where available.

### `mask_covidx_overlays.py`

Applies clip-specific polygon masks to the staged PNGs for scanner text, icons, depth markings, and watermarks. It generates side-by-side previews under `data/intermediate/covidx_us/shortlist/overlay_masks/previews/` so we can check the overlay and the scan edge together. With `--apply`, it points `frames.csv` to the masked copies and records the mask polygons in output-pixel coordinates. Use those paths in `approved.csv`.

## Katumba pipeline

### `build_katumba_manifest.py`

Reads `data/raw/katumba/metadata.xlsx` and the processed images under `data/raw/katumba/dataset/`. It matches filenames to workbook rows, maps `Healthy Lung` to `healthy` and `Probably Covid` to `covid`, and flags out-of-scope labels and disagreements for exclusion from staging. It pools the original train, validation, and test folders by source patient ID and retains the original split for provenance. Writes `data/intermediate/katumba/intake_manifest.csv`.

### `stage_katumba_stills.py`

Reads the eligible intake rows, removes exact pixel duplicates within each group, and selects up to three stills spread across filename order. It designates one default still per group, converts the selected stills to grayscale, and resizes and pads them proportionally to 512 × 512. Writes PNGs and `staged_manifest.csv` under `data/intermediate/katumba/staged/`.

### `audit_katumba_overlap.py`

Compares staged images against the shared clean export and against images assigned to other Katumba groups. Exact pixel matches and perceptual hash similarities go to `data/intermediate/katumba/overlap_audit.csv`. Similarity flags are candidates for review. Patient grouping depends on source metadata and findings from the review.

### `resolve_katumba_duplicate.py`

Verifies the three one-to-one image matches between the two audited Katumba COVID groups. It retains the original train group and excludes its three validation copies from `staged/resolved_staged_manifest.csv`. The retained group can go into any subsequent project split.

### `review_katumba_overlap.py`

Uses the resolved manifest to remove audit pairs involving excluded stills. Writes `data/intermediate/katumba/overlap_review.csv` and reports which comparisons need review. In the published batch, 49 cross-source perceptual matches were auto-cleared based on separate cohort provenance, and all 44 pending within-Katumba image pairs were cleared after side-by-side review.

### `publish_katumba.py`

Checks the resolved manifest and overlap report against the reviewed batch, records the 44 decisions, and publishes 187 stills across 64 provisional groups. It adds or refreshes Katumba rows in the shared manifest and records the workbook label and original source split in `original_label` and `source_split`. Full-frame Katumba exports retain `needs_crop_review`. This publisher is tied to the reviewed release. Review any changed batch and update the publisher checks before publishing it.

## Rebuild the combined export

Run from the repository root with pandas, NumPy, OpenCV, and Pillow installed and the original metadata and media in `data/raw/`.

```powershell
python .\data_support\build_recording_manifest.py
python .\data_support\extract_baseline_stills.py
python .\data_support\normalize_stills.py
python .\data_support\integrate_covidx_shortlist.py
python .\data_support\mask_covidx_overlays.py
```

Review the mask previews, then apply the masks and place the reviewed `approved.csv` beside `frames.csv` under `data/intermediate/covidx_us/shortlist/`. Run the last command after the approval paths match the updated staged manifest.

```powershell
python .\data_support\mask_covidx_overlays.py --apply
python .\data_support\integrate_covidx_shortlist.py --publish
```

After publishing COVIDx-US, run the Katumba scripts in this order for the reviewed release.

```powershell
python .\data_support\build_katumba_manifest.py
python .\data_support\stage_katumba_stills.py
python .\data_support\audit_katumba_overlap.py
python .\data_support\resolve_katumba_duplicate.py
python .\data_support\review_katumba_overlap.py
python .\data_support\publish_katumba.py
```

The completed export contains 471 stills across 179 provisional patient groups.

Run the overlap audit again when the source inventory changes. Keep raw media and working files locally under `data/raw/` and `data/intermediate/`. The combined export lives in `data/clean/images/` and `data/clean/manifests/normalization_manifest.csv`.

## Important Notes for Training Models

> [!IMPORTANT]
> **Split by patient group.** Use `group_id` to keep all views of the same provisional patient group together in one train, validation, or test split. Generate project splits from `group_id` and use `source_split` for Katumba provenance. Load images using `normalized_path` relative to the repository root.

> [!CAUTION]
> **Count patient groups when reporting cohort size.** A group can contribute one to three stills. Use `is_default` for a one-still-per-group baseline.
