# AI Based Lung Disease and Detection from Ultrasound Data

BMEG 457 Group 2 capstone project. We are building a lung ultrasound classifier for three classes: healthy, pneumonia, and COVID-19.

## Datasets and preprocessing

The first dataset in this repository is [Born POCUS](https://github.com/jannisborn/covid19_ultrasound), a collection of lung ultrasound images and videos. We map the source labels `regular` to **healthy**, `Bacterial pneumonia` and `Viral pneumonia` to **pneumonia**, and `COVID-19` to **COVID-19**.

Additional Datastets will be added as they are normalized to the same universal standard documented below. 

### Born POCUS Dataset Summaries

| Class | Matched recordings | Candidate recordings | Provisional patient groups | Normalized stills |
|---|---:|---:|---:|---:|
| Healthy | 69 | 68 | 41 | 103 |
| Pneumonia | 57 | 56 | 52 | 128 |
| COVID-19 | 40 | 40 | 16 | 44 |
| **Total** | **166** | **164** | **109** | **275** |


| Class | Provisional patient groups | Share of groups | Groups with 1 still | Groups with 3 stills |
|---|---:|---:|---:|---:|
| Healthy | 41 | 37.6% | 10 | 31 |
| Pneumonia | 52 | 47.7% | 14 | 38 |
| COVID-19 | 16 | 14.7% | 2 | 14 |
| **Total** | **109** | **100.0%** | **26** | **83** |

COVID-19 is the smallest class, with 16 of 109 provisional patient groups. Use these group counts when checking class balance in the train/test split.

The source metadata listed 171 entries as hosted. We matched 166 to files in the main image and video folders and excluded two matched recordings that shared a patient key but had conflicting labels. Of the 109 remaining provisional patient groups, 83 contributed three stills and 26 contributed one.

## Harmonized Image Standard 

This section outlines the general pre-processing steps used to achieve the harmonized image standard. We have designed it to be scalable across a diversit of datasets, allowing the training dataset to continuously grow in parallel with model development 

In a nutshell, all of our training data has the following characteristics: 

Each dataset added to this repository follows the same output standard.

| Characteristic | Standard |
|---|---|
| Image format | Single-channel, 8-bit grayscale PNG. Each file contains one lung ultrasound still, either supplied as an image or extracted from a video. |
| Image dimensions | 224 × 224 pixels, including padding. The scan is resized proportionally and centered on a black canvas. |
| Class labels | Each image has one harmonized label in the manifest: `healthy`, `pneumonia`, or `covid`. |
| Stills per patient group | Each group contributes one to three stills, with exactly one marked `is_default=True`. All stills from the same patient share a `group_id` and stay in the same dataset split. |
| Traceability | Each image has one manifest row linking it to its source dataset and recording, frame index where applicable, and normalized file path relative to the repo root. |
| Crop review | The manifest records crop coordinates and review status. Images without a configured crop retain the full frame and are flagged for review. |

The 224 × 224 standard above describes the output canvas. Physical scan depth, field of view, and native image resolution can still vary by source.

### Universal Pre-Processing Steps

1. Match source metadata to media files, harmonize labels, and group recordings using the source and patient identifiers.
2. Select up to three stills per group. Prefer existing stills and views from distinct clips, then sample additional frames from the same clip when needed. Skip exact pixel duplicates within each group and designate one still as the default.
3. Apply a crop where a rule is configured, convert each still to grayscale, and resize it proportionally onto a black **224 × 224** canvas. Save the result as a PNG.

The normalized images and `normalization_manifest.csv` are in [`data/clean/born_pocus/`](data/clean/born_pocus/). Load images using `normalized_path` relative to the repository root. The manifest also records the class, provisional patient group, default-still flag, source recording, frame index when applicable, and crop-review status. See [`data_support/`](data_support/) for pipeline details.

### Normalization manifest

The structure of the normalization_manifest.csv is defined here. 

`data/clean/born_pocus/normalization_manifest.csv` contains one row per normalized still. Load images using `normalized_path` and keep all rows with the same `group_id` in the same dataset split.

| Column | Meaning |
|---|---|
| `group_id` | Provisional patient group ID, such as `born_0001`. All stills assigned to the same group share this ID. |
| `class` | Harmonized label. Values are `healthy`, `pneumonia`, or `covid`. |
| `still_number` | Still number within the group, from 1 to 3. |
| `is_default` | `True` for the designated still to use in a one-image-per-group baseline. |
| `source_path` | Path to the original image or video relative to `data/raw/born_pocus/`. |
| `frame_index` | Zero-based frame index for a still extracted from a video. Blank for an original still. |
| `origin` | How the still was obtained. Values are `original_still` or `video_frame`. |
| `additional_frame_from_same_clip` | `True` if an earlier still in the same group came from the same source clip. |
| `image_path` | Repo-relative path to the intermediate still before normalization. Used for local auditing. |
| `status` | Extraction status carried forward from the multi-view manifest. `review` means the still was saved for review. |
| `source_id` | Born POCUS metadata identifier for the original contributing source. |
| `probe` | Ultrasound probe type recorded in the source metadata, when available. |
| `dataset` | Dataset the still came from. The current value is `born_pocus`. |
| `original_width` | Width of the still before cropping or resizing, in pixels. |
| `original_height` | Height of the still before cropping or resizing, in pixels. |
| `crop_x0` | Left edge of the retained region, in original-image pixels, inclusive. |
| `crop_y0` | Top edge of the retained region, in original-image pixels, inclusive. |
| `crop_x1` | Right edge of the retained region, in original-image pixels, exclusive. |
| `crop_y1` | Bottom edge of the retained region, in original-image pixels, exclusive. |
| `output_width` | Width of the final PNG canvas, including padding. Currently 224 pixels. |
| `output_height` | Height of the final PNG canvas, including padding. Currently 224 pixels. |
| `normalized_path` | Repo-relative path to the normalized PNG. Use this column to load the training image. |
| `normalization_qc` | `review_crop` means a configured crop was applied and awaits review. `needs_crop_review` means the full frame was used and awaits review. |
