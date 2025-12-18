# Dataset Download & Setup

This project uses **publicly available datasets** for car damage detection and classification.

Due to dataset size and licensing restrictions, **raw image data is not included** in this repository.
This document explains how to download and organize the datasets locally so all experiments are reproducible.

---

## 1. CarDD (Car Damage Dataset)

**Purpose**
- Used for:
  - Damage vs no-damage detection (positive samples)
  - Damage type classification

**Paper**
- CarDD: A New Dataset for Vision-based Car Damage Detection  
- IEEE (2022)

**Download**
- Official repository or dataset page:
  - https://github.com/CarDD-USTC/CarDD

**Expected Classes**
- dent
- scratch
- crack
- glass_shatter
- lamp_broken
- tire_flat

**Setup**
After downloading and extracting:

data/raw/car_dd/
├── images/
│ ├── dent/
│ ├── scratch/
│ ├── crack/
│ ├── glass_shatter/
│ ├── lamp_broken/
│ └── tire_flat/
└── annotations/ # (if provided, not required for v1)

Only the image folders are required for this project.

---

## 2. Stanford Cars Dataset

**Purpose**
- Used as **no-damage (clean car)** examples for the binary damage detection model.

**Description**
- Contains high-quality images of cars across many makes and models.
- Assumed to represent undamaged vehicles.

**Download**
- Official dataset page:
  - https://ai.stanford.edu/~jkrause/cars/car_dataset.html

**Setup**
After extracting:

data/raw/stanford_cars/
├── car_ims/
│ ├── 00001.jpg
│ ├── 00002.jpg
│ └── ...
└── devkit/

Only the `car_ims/` directory is required.

---

## 3. Directory Structure (Required)

Your local directory structure **must** look like this:

data/
├── raw/
│ ├── car_dd/
│ └── stanford_cars/
├── processed/
└── splits/

---

## 4. Notes & Assumptions

- Raw datasets are **never committed** to GitHub.
- Train/validation/test splits are generated via `scripts/make_splits.py`.
- Stanford Cars images are treated as **no-damage** samples.
- This project focuses on **image-level classification**, not localization or segmentation.

---

## 5. Verification

Before proceeding, verify:
- Images can be loaded with PIL/OpenCV
- Class folders exist for all CarDD categories
- No images are accidentally committed to git

You are now ready to generate dataset splits and begin training.