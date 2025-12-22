# 🚗 Car Damage Detection & Classification Service

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![Inference](https://img.shields.io/badge/Inference-CPU--only-lightgrey)
![Status](https://img.shields.io/badge/Status-Research%20%2F%20Prototype-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A production-style computer vision service that detects whether a car image contains damage and optionally classifies the type of damage.

The system is designed as a **first-pass automated screening tool**, suitable for use cases such as insurance intake, rental inspection, or used car listings.

---

## 📌 Project Overview

**Input:** Image of a car  
**Output:**
- Damage detected: `true / false`
- Damage confidence score
- Damage type (if damaged)
- Inference latency (ms)

The service exposes a **REST API** built with **FastAPI** and is optimized for **CPU-only inference**.

---

## 🧠 System Design

### High-Level Architecture

```
Client
  |
  | POST /detect-damage
  v
FastAPI Service
  ├── Image validation & preprocessing
  ├── Binary damage detector (damage / no-damage)
  ├── Damage type classifier (if damaged)
  └── JSON response with confidences & latency
```

---

## 📊 Dataset

This project uses two public datasets:

### 1️⃣ CarDD (Car Damage Dataset)
- Images of damaged cars with labeled damage types
- Damage classes:
  - dent
  - scratch
  - crack
  - glass_shatter
  - lamp_broken
  - tire_flat

### 2️⃣ Stanford Cars Dataset
- Used as **no-damage** examples
- Clean car images across many makes and models

📌 **Important Note**  
Raw datasets are **not included** in this repository due to size and licensing constraints.  
See [`scripts/download_data.md`](scripts/download_data.md) for dataset setup instructions.

---

## 🧪 Machine Learning Approach

### Modeling Strategy
- **Two-stage classification pipeline**
  1. Binary classifier: `damage` vs `no-damage`
  2. Multi-class classifier: damage type (only if damage is detected)

### Model Architecture
- Pretrained CNN backbone (e.g., MobileNetV3 / ResNet18)
- Transfer learning from ImageNet weights
- Optimized for low-latency CPU inference

### Training Details
- Image size: **224 × 224**
- Lightweight augmentations
- Early stopping and threshold tuning
- Metrics:
  - Precision
  - Recall
  - F1-score
  - Confusion matrices

---

## 📈 Results

> Results will vary depending on dataset split and training configuration.

**Binary Damage Detection**
- High recall prioritized to minimize false negatives
- Stable inference latency on CPU (~30–40 ms)

**Damage Type Classification**
- Strong performance on common damage classes (dent, scratch)
- Reduced accuracy on visually ambiguous classes (crack vs scratch)

📌 Detailed evaluation artifacts (metrics, plots) are saved during training and can be found in the `models/` directory.

---

## 🚀 API Usage

### Endpoint
```
POST /detect-damage
```

### Request
- Content-Type: `multipart/form-data`
- Field: `file` (image)

### Example Response (No Damage)
```json
{
  "damage_detected": false,
  "damage_confidence": 0.91,
  "damage_type": null,
  "type_confidence": null,
  "latency_ms": 38
}
```

---

## 🛠️ Running Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Download datasets
Follow instructions in:
```text
scripts/download_data.md
```

### 3️⃣ Generate dataset splits
```bash
python scripts/make_splits.py
```

### 4️⃣ Train models
```bash
python scripts/train_binary.py
python scripts/train_type.py
```

### 5️⃣ Start the API
```bash
uvicorn app.main:app --reload
```

---

## 📁 Repository Structure

```
car-damage-detection-service/
├── app/            # FastAPI service
├── data/           # Dataset splits (raw data excluded)
├── models/         # Trained model artifacts
├── scripts/        # Training and preprocessing scripts
├── tests/          # Unit and API tests
├── Dockerfile
├── .gitignore
└── README.md
```

📌 **Note:**  
Intermediate artifacts (logs, `.txt` notes, and local experiment files) are excluded via `.gitignore`.

---

## ⚠️ Limitations & Future Work

### Known Limitations
- Performance may degrade under extreme lighting or heavy occlusions
- Domain shift between datasets may impact generalization
- No localization of damage (classification only)

### Planned Improvements
- Damage localization (bounding boxes)
- Segmentation-based severity estimation
- Model quantization for faster CPU inference
- Batch inference support
- Model versioning and monitoring

---

## 📜 License
This project is intended for **educational and research purposes**.  
Please verify dataset licenses before commercial use.
