# Dataset Preprocessing

These scripts prepare lung ultrasound stills and frames from Born POCUS, COVIDx-US, and Katumba for the capstone’s three-class classifier. The shared dataset uses `healthy`, `pneumonia`, and `covid` labels.

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
| [`publish_katumba.py`](publish_katumba.py) | Builds the reviewed Katumba subset for the capped baseline and applies the positive rapid-test filter. | Preserves the label decision and archives previously published COVID stills placed on hold. |
| [`expand_dataset_frames.py`](expand_dataset_frames.py) | Decodes eligible media across all three sources and removes exact duplicates after normalization. | Builds the current export without a fixed frame cap. |

## Born POCUS pipeline

### `build_recording_manifest.py`

Reads the Born POCUS metadata under `data/raw/born_pocus/` and matches it to the local image and video files. It maps the original labels to the project classes and uses the source and patient identifiers to create provisional groups. It records unmatched and ambiguous files for review, and excludes groups with conflicting labels from the candidate list.

Outputs `recordings.csv` and `baseline_candidates.csv` under `data/intermediate/born_pocus/`. The recording inventory provides the provenance used by the COVIDx-US overlap audit.

### `extract_baseline_stills.py`

Reads the candidate list and selects up to three stills per group. It prefers existing stills and distinct source clips, then samples additional frames if needed. It skips exact pixel duplicates within a group and designates one still as the default. The multi-view manifest records the source path and frame index so a selected still can be traced back to its recording.

Writes the stills and `multi_view_manifest.csv` under `data/intermediate/born_pocus/`.

### `normalize_stills.py`

Reads the Born POCUS multi-view manifest and applies configured crops, converts the stills to 8-bit grayscale, and rescales them proportionally. It upscales or downscales as needed, then centers each image on a black 512 × 512 canvas. The fixed canvas gives model inputs a consistent shape. Native resolution, field of view, and physical scan depth can vary by source.

The default run builds the Born POCUS clean export. Use `--pilot` to inspect a sample of before-and-after images. Use the default run when reconstructing the capped seed export from source inputs. The optional `--include-covidx-drafts` mode serves draft review.

## COVIDx-US pipeline

### `audit_covidx_overlap.py`

Reads the COVIDx-US video metadata and compares upstream URLs against the matched Born POCUS recording inventory across all labels. It distinguishes direct clip URL matches from shared case-page links and flags possible label conflicts. It also identifies COVIDx-US patient groups linked to a direct match.

Writes `pneumonia_overlap_audit.csv`, `covid_overlap_audit.csv`, and `normal_overlap_audit.csv` under `data/intermediate/covidx_us/`. Follow up with source case descriptions and clip comparisons where needed. Different URLs can still identify the same recording or patient. A normal-looking scan also needs case evidence before it is assigned the `healthy` label.

### `extract_covidx_stills.py`

Samples frames at approximately 25%, 50%, and 75% of each configured clip in the initial pneumonia review batch. It skips exact duplicate frames within a clip and records each sampled frame in `draft_stills_manifest.csv`. The PNGs and CSV stay under `data/intermediate/covidx_us/`.

Use this script when revisiting the initial candidate review. The approved shortlist has its own staging and export workflow below.

### `integrate_covidx_shortlist.py`

Reads the shortlisted videos from `data/raw/covidx_us/videos/` and stages sample frames under `data/intermediate/covidx_us/shortlist/`. It applies a clip-specific crop where needed, including the crop that isolates a lung panel from the combined heart and lung views in clip `174`. It uses the same 512 × 512 normalization as Born POCUS. The `frames.csv` inventory records the class, patient group, source clip, frame index, and crop bounds. Two clips from one case can share a group, as with Kalafat clips `199` and `200`.

After image and mask review, `--publish` reads the selected paths from `approved.csv`, enforces a maximum of three stills per group, and builds the Born POCUS and COVIDx-US seed manifest. Apply this step before publishing Katumba when reconstructing the seed export. It expects the approved CSV to contain the full COVIDx-US selection intended for export. Keep optional source reuse references in its `reuse_evidence` column where available.

### `mask_covidx_overlays.py`

Applies clip-specific polygon masks to the staged PNGs for scanner text, icons, depth markings, and watermarks. It generates side-by-side previews under `data/intermediate/covidx_us/shortlist/overlay_masks/previews/` so we can check the overlay and the scan edge together. With `--apply`, it points `frames.csv` to the masked copies and records the mask polygons in output-pixel coordinates. Use those paths in `approved.csv`.

## Katumba pipeline

### `build_katumba_manifest.py`

Reads `data/raw/katumba/metadata.xlsx` and the processed images under `data/raw/katumba/dataset/`. It matches filenames to workbook rows, maps `Healthy Lung` to `healthy` and records `Probably Covid` as a COVID candidate for downstream evidence review, and flags out-of-scope labels and disagreements for exclusion from staging. It pools the original train, validation, and test folders by source patient ID and retains the original split for provenance. Writes `data/intermediate/katumba/intake_manifest.csv`.

### `stage_katumba_stills.py`

Reads the eligible intake rows, removes exact pixel duplicates within each group, and selects up to three stills spread across filename order. It designates one default still per group, converts the selected stills to grayscale, and resizes and pads them proportionally to 512 × 512. Writes PNGs and `staged_manifest.csv` under `data/intermediate/katumba/staged/`.

### `audit_katumba_overlap.py`

Compares staged images against the shared clean export and against images assigned to other Katumba groups. Exact pixel matches and perceptual hash similarities go to `data/intermediate/katumba/overlap_audit.csv`. Similarity flags are candidates for review. Patient grouping depends on source metadata and findings from the review.

### `resolve_katumba_duplicate.py`

Verifies the three one-to-one image matches between the two audited Katumba COVID groups. It retains the original train group and excludes its three validation copies from `staged/resolved_staged_manifest.csv`. The retained group can go into any subsequent project split.

### `review_katumba_overlap.py`

Uses the resolved manifest to remove audit pairs involving excluded stills. Writes `data/intermediate/katumba/overlap_review.csv` and reports which comparisons need review. In the published batch, 49 cross-source perceptual matches were auto-cleared based on separate cohort provenance, and all 44 pending within-Katumba image pairs were cleared after side-by-side review.

### `publish_katumba.py`

Checks the resolved staging manifest and overlap report for the reviewed baseline. It publishes 117 healthy stills from 40 groups and 11 COVID stills from four groups whose workbook symptom history explicitly reports a positive rapid diagnostic test. It moves 59 previously published COVID stills from 20 held groups out of the clean image directory and into `data/intermediate/katumba/label_hold/images/` when those files are present. The manifest retains the original workbook conclusion and source split. Test-to-scan timing remains under review.

## Expanded export pipeline

### `expand_dataset_frames.py`

Reads the Born POCUS candidate inventory, the current clean manifest for the approved COVIDx-US shortlist and retained Katumba groups, and the Katumba intake manifest and workbook. It also needs the corresponding raw images, videos, and GIFs. It imports helpers and rules from `normalize_stills.py`, `mask_covidx_overlays.py`, and `build_katumba_manifest.py`.

Decodes one candidate per eligible still image and every decodable frame from eligible videos and animated GIFs within applicable scan restrictions. It applies crops, masks, and 512 × 512 grayscale normalization before comparing pixels for exact duplicates across the export. The manifest retains one default still per group. Skipped frames and their retained matches are listed in `manifests/exact_duplicates.csv`.

COVIDx-US clip `174_core_covid` contributes the previously approved frames at zero-based indices 38 and 57. Supply `--lung-range-174 FIRST LAST` only when a continuous lung-only range has been verified. Both endpoints are included. Newly extracted COVIDx-US frames are flagged `needs_frame_review`.

| Source | Stills and frames | Provisional patient groups |
|---|---:|---:|
| Born POCUS | 20,033 | 109 |
| COVIDx-US | 1,235 | 6 |
| Katumba | 405 | 44 |
| **Total** | **21,673** | **159** |

Katumba contributes 362 healthy images and 43 COVID images. The published build skipped 20 exact normalized duplicates, all within patient groups.

## Rebuild the combined export

Run from the repository root with pandas, NumPy, OpenCV, and Pillow installed and the original metadata and media in `data/raw/`.

### Expand an existing combined export

If the current clean manifest includes all three sources and the candidate inventories are present, run the expansion directly.

```powershell
python .\data_support\expand_dataset_frames.py
```

The preview writes to `data/intermediate/expanded_export/`. To build and publish a fresh expansion, run:

```powershell
python .\data_support\expand_dataset_frames.py --publish
```

Publication archives the previous `data/clean/` under `data/intermediate/` and prints the backup path. Use the same verified clip 174 range for both commands if you supply one.

### Reconstruct the reviewed seed export

Use the following sequence when starting from the original source inputs and saved approvals.

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

The reviewed seed export contains 412 stills across 159 provisional patient groups. Run the expansion command above to publish the full frame export.

Run the overlap audit again when the source inventory changes. Keep raw media and working files locally under `data/raw/` and `data/intermediate/`. The combined export lives in `data/clean/images/` and `data/clean/manifests/normalization_manifest.csv`.

## Important Notes for Training Models

> [!IMPORTANT]
> **Split by patient group.** Use `group_id` to keep all views of the same provisional patient group together in one train, validation, or test split. Generate project splits from `group_id` and use `source_split` for Katumba provenance. Load images using `normalized_path` relative to the repository root.

> [!CAUTION]
> **Count patient groups when reporting cohort size.** The expanded export has no fixed frame cap per group. Use `is_default` for a one-still-per-group baseline.
