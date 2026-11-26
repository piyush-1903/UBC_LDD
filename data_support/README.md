# Dataset Preprocessing

This folder contains the scripts that prepare lung ultrasound stills for the capstone’s three-class classifier: healthy, pneumonia, and COVID-19. 

The current pipeline processes Born POCUS. I (Piyush) am working on additional datasets that will standardize to the outputs and labelling used by BornPOCUS. 

The pre-processing pipeline is as follows:

1. **Build the recording manifest.** Match source metadata to local images and videos, map the original labels to the three project classes, and group recordings using source and patient identifiers. Exclude unmatched files and patient groups with conflicting labels from the candidate set.
2. **Select up to three stills per patient group.** Use existing stills and frames from distinct clips first, then try additional frames from the same clip when needed. Skip exact duplicates within each group and mark one still as the default.
3. **Standardize the images.** Apply a crop where a rule is configured, convert to grayscale, and resize proportionally onto a black 224 × 224 canvas. Images without a configured crop are flagged for review.

The standardized output is a set of **224 × 224 grayscale PNGs** and a **`normalization_manifest.csv`**, saved under `data/clean/born_pocus/`. The manifest records each image’s class, provisional patient group, default-still flag, source recording, frame index when applicable, and crop-review status. Use `normalized_path` to load images and `group_id` to keep all views from a patient group together when splitting the dataset.


# Important Notes for Training Models 

> [!IMPORTANT]
> **Split by patient group.** Use `group_id` to keep all stills from the same
> patient group in the same train, validation, or test split. Load files using
> `normalized_path`, which is relative to the repository root.

> [!CAUTION]
> **Each Image is not a patient!** Each image is not one patient. When possible, 
> I have provided up to 3 image stills per patient. This may not be possible with 
> all future datasets. 


Original media stays in `data/raw/` in my (Piyush's) machine to prevent bloating. Working files and pilot previews stay in `data/intermediate/` on the same machine to prevent bloating.