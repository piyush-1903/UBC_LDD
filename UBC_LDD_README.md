# AI Based Lung Disease and Detection from Ultrasound Data

BMEG 457 Group 2 capstone project. We are building a lung ultrasound classifier for three classes: healthy, pneumonia, and COVID-19.

## Datasets and preprocessing

### General Dataset

We built a preprocessing pipeline that combines lung ultrasound recordings from multiple sources into a shared dataset for training and evaluation. For each source, we match metadata to the original images and videos, map labels to healthy, pneumonia, or COVID-19, and group recordings using the available patient or case identifiers. When adding a dataset, we check for overlap with existing recordings and review ambiguous labels before selecting stills.

Each included patient group contributes up to three stills, with one designated as the default. We apply crops and overlay masks where needed, then export the stills as **512 × 512 grayscale PNGs** with aspect ratio preserved. A shared manifest records each still’s class, source, patient group, and preprocessing details.

The combined dataset has the following structure: 

| Class | Born POCUS patient groups | COVIDx-US patient groups | Katumba patient groups | Total patient groups | Born POCUS stills | COVIDx-US stills | Katumba stills | Total stills |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Healthy | 41 | 1 | 40 | 82 | 103 | 1 | 117 | 221 |
| Pneumonia | 52 | 3 | 0 | 55 | 128 | 4 | 0 | 132 |
| COVID-19 | 16 | 2 | 24 | 42 | 44 | 4 | 70 | 118 |
| **Total** | **109** | **6** | **64** | **179** | **275** | **9** | **187** | **471** |

Patient groups are provisional and are based on the available source patient or case identifiers. Each group can contribute up to three stills.

It draws from the following datasets.

- Born POCUS
- COVIDx-US
- [Katumba lung ultrasound dataset, version 2](https://data.mendeley.com/datasets/hb3p34ytvx/2)

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

The source metadata listed 171 entries as hosted. We matched 166 to files in the main image and video folders and excluded two matched recordings that shared a patient key but had conflicting labels. Of the 109 remaining provisional patient groups, 83 contributed three stills and 26 contributed one.

### COVIDx-US Dataset Summary

| Class | Source clips audited | Source clips included | Provisional patient groups | Normalized stills |
|---|---:|---:|---:|---:|
| Healthy | 28 | 1 | 1 | 1 |
| Pneumonia | 49 | 3 | 3 | 4 |
| COVID-19 | 72 | 3 | 2 | 4 |
| **Total** | **149** | **7** | **6** | **9** |

| Class | Provisional patient groups | Share of groups | Groups with 1 still | Groups with 2 stills | Groups with 3 stills |
|---|---:|---:|---:|---:|---:|
| Healthy | 1 | 16.7% | 1 | 0 | 0 |
| Pneumonia | 3 | 50.0% | 2 | 1 | 0 |
| COVID-19 | 2 | 33.3% | 0 | 2 | 0 |
| **Total** | **6** | **100.0%** | **3** | **3** | **0** |

We audited 149 COVIDx-US clips across the three target classes. The release includes **seven source clips from six provisional patient groups**. The two Kalafat clips belong to one patient group. The seven clips yielded 21 draft stills, of which we selected nine.

- **Included - 7 clips.** These represent three pneumonia groups, two COVID-19 groups, and one documented healthy control.
- **On hold - 89 clips.** The available evidence left questions about the clip-to-case link, patient identity, overlap with Born POCUS, or whether the source supported the target label.
- **Excluded or deferred - 53 clips.** This includes recordings already represented in Born POCUS, clips with unsuitable anatomy or labels, and candidates deferred from this release. Examples include the five Volpicelli clips already represented in Born POCUS and normal-side comparison views from patients with pneumothorax.
- **Outside audit scope - 94 entries.** COVIDx-US labels these `Other`. They are outside the 149 clips audited for the three target classes.

The audit totals count source clips. Patient-group totals are reported for the included clips, where the available case information supports grouping. 

Clips that are on hold will be undergoing a manual review and source attributino, before being merged or rejected from the master dataset.  

### Katumba Dataset Summary

| Class | Eligible source images | Candidate patient groups | Published patient groups | Normalized stills |
|---|---:|---:|---:|---:|
| Healthy | 362 | 40 | 40 | 117 |
| COVID-19 | 325 | 25 | 24 | 70 |
| **Total** | **687** | **65** | **64** | **187** |

The intake manifest indexed 1,062 source images. The source workbook labels determined eligibility. We mapped `Healthy Lung` to `healthy` and `Probably Covid` to `covid`. The other 375 images were outside the selected categories or had a folder label that disagreed with the workbook. This includes 13 images in the source COVID folder that the workbook marked as probably not COVID.

Staging skipped 10 exact duplicates within patient groups and selected 190 stills. The overlap audit found three exact image pairs linking the same two Katumba COVID groups. We kept one copy of each still under a single provisional group. The visual review cleared the 44 remaining within-Katumba image pairs, and the audit found zero exact image matches with Born POCUS or COVIDx-US.

We pooled Katumba’s original train, validation, and test sets by patient group. The manifest retains the original split for provenance. Generate project splits using `group_id` so all stills from each group stay together.

## Harmonized Image Standard

We use a common output standard for lung ultrasound stills from all source datasets. This lets us add new recordings without changing how the training pipeline loads images, reads labels, or groups patient views.

| Characteristic | Standard |
|---|---|
| Image format | Single-channel, 8-bit grayscale PNG. Each file contains one lung ultrasound still, either supplied as an image or extracted from a video. |
| Image dimensions | 512 × 512 pixels, including padding. The image is resized proportionally and centered on a black canvas. |
| Class labels | Each image has one harmonized label in the manifest: `healthy`, `pneumonia`, or `covid`. |
| Stills per patient group | Each provisional group contributes one to three stills, with exactly one marked `is_default=True`. Keep all stills with the same `group_id` in the same dataset split. |
| Cropping and overlay masking | Apply configured crops to isolate lung views and masks to remove distracting overlays where feasible. The manifest records crop bounds, mask polygons where applied, and review status. |
| Traceability | Each still has one row in the shared manifest with its source dataset, recording, frame index where applicable, and repo-relative `normalized_path`. |

The 512 × 512 dimensions describe the exported canvas. Native image resolution, physical scan depth, and field of view still vary by source.

### Universal Pre-Processing Steps

The pipeline applies the following techniques as needed for each source dataset:

- **Match metadata to media.** Link source records to the original images or videos, map source labels to the three project classes, and group recordings using available patient or case identifiers.
- **Check for overlap and label conflicts.** Compare source URLs, clip identifiers, and case descriptions against recordings already in the dataset. Flag unresolved matches or labels for review.
- **Extract and select stills.** Use existing stills or extract frames from videos. Prefer distinct views, skip exact pixel duplicates within a patient group, and select up to three stills per group. Mark one as the default.
- **Crop and mask where needed.** Crop to isolate the relevant lung view when a frame contains multiple panels. Apply targeted masks to scanner text, icons, depth markings, or watermarks where feasible.
- **Convert to grayscale.** Standardize the output to single-channel, 8-bit PNGs.
- **Upscale or downscale.** Resize proportionally to fit a **512 × 512** canvas. Larger images are downscaled, and smaller images are upscaled when needed.
- **Pad to a common size.** Center the resized image on a black canvas to fill the remaining space while preserving the scan’s aspect ratio.
- **Export images and record provenance.** Save the PNGs in `data/clean/images/` and the shared manifest at `data/clean/manifests/normalization_manifest.csv`. The manifest records the source, label, patient group, frame index where applicable, and crop or mask details.

See the [data support README](data_support/README.md) for what each script does, why each step is used, and the order in which to run them.


### Normalization manifest

[`data/clean/manifests/normalization_manifest.csv`](data/clean/manifests/normalization_manifest.csv) contains one row per exported still from Born POCUS, COVIDx-US, or Katumba. Load images using `normalized_path` relative to the repository root. Keep all rows sharing a `group_id` in the same train, validation, or test split. Some provenance fields are populated only where the source provides them.

| Column | Meaning |
|---|---|
| `group_id` | Provisional patient group ID. All stills assigned to the same group share this ID. |
| `group_scope` | Basis of the grouping. `source_patient` uses source patient or case information. `clip_only`, if present in a draft export, groups frames by clip. |
| `class` | Harmonized label. Values are `healthy`, `pneumonia`, or `covid`. |
| `dataset` | Source dataset. Values are `born_pocus`, `covidx_us`, and `katumba`. |
| `still_number` | Still number within the group, from 1 to 3. |
| `is_default` | `True` for the designated still to use in a one-image-per-group baseline. |
| `source_id` | Identifier or name of the contributing source, depending on the dataset. |
| `source_patient_id` | Original patient identifier, where recorded. |
| `clip_id` | COVIDx-US clip identifier, where recorded. |
| `source_url` | Upstream media URL, where recorded. |
| `source_path` | Path to the original media relative to the source dataset’s raw folder. |
| `frame_index` | Zero-based frame index for a still extracted from video. Blank for an original still. |
| `origin` | How the still was obtained. Values are `original_still` or `video_frame`. |
| `additional_frame_from_same_clip` | `True` if an earlier selected still in the same group came from the same clip. |
| `image_path` | Repo-relative path to the intermediate image used for local review or auditing. For masked stills, it points to the masked staging copy. |
| `normalized_path` | Repo-relative path to the exported PNG in `data/clean/images/`. Use this column to load the image. |
| `probe` | Ultrasound probe type when available. |
| `original_width`, `original_height` | Image dimensions before cropping or resizing, in pixels. |
| `crop_x0`, `crop_y0` | Inclusive left and top edges of the retained region, in original-image pixels. |
| `crop_x1`, `crop_y1` | Exclusive right and bottom edges of the retained region, in original-image pixels. |
| `output_width`, `output_height` | Final canvas dimensions, including padding. Both are currently 512 pixels. |
| `overlay_mask_polygons` | JSON-encoded polygon coordinates in output-image pixels. Populated where an overlay mask was applied. |
| `status` | Stage or selection status. `review` is carried forward from still extraction, and `approved` identifies selected COVIDx-US and Katumba export rows. |
| `normalization_qc` | Review state. `review_crop` flags a configured crop for review, `needs_crop_review` flags a full-frame export for review, and `approved` records approval of the selected still. |
| `original_label` | Original Katumba workbook label before harmonization. |
| `source_split` | Original Katumba split for provenance. Generate project splits using `group_id`. |
| `reuse_evidence` | Optional reference documenting reuse terms, where recorded. |
