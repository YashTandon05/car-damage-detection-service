# Car Damage Detection & Classification Service

A production-style computer vision service that detects whether a car image contains damage and optionally classifies the type of damage.

The system is designed as a **first-pass automated screening tool**, suitable for use cases such as insurance intake, rental inspection, or used car listings.

---

## 🚗 Project Overview

**Input:** Image of a car  
**Output:**
- Damage detected: yes / no
- Damage confidence score
- Damage type (if damaged)
- Inference latency

The service exposes a **REST API** built with FastAPI and runs entirely on **CPU**.

---

## 🧠 System Design

### High-level Architecture

Client
|
| POST /detect-damage
v
FastAPI Service
├── Image validation & preprocessing
├── Binary damage detector (damage / no-damage)
├── Damage type classifier (if damaged)
└── JSON response with confidences

---

## 📊 Dataset

This project uses two public datasets:

### 1. CarDD (Car Damage Dataset)
- Provides images of damaged cars with labeled damage types
- Classes:
  - dent
  - scratch
  - crack
  - glass_shatter
  - lamp_broken
  - tire_flat

### 2. Stanford Cars
- Used as **no-damage** examples
- Contains clean images of cars across many makes and models

📌 **Note:**  
Raw datasets are **not included** in this repository due to size and licensing constraints.  
See [`scripts/download_data.md`](scripts/download_data.md) for setup instructions.

---

## 🧪 Machine Learning Approach

### Modeling Strategy
- **Two-stage classification**
  1. Binary classifier: damage vs no-damage
  2. Multi-class classifier: damage type (only if damage is detected)

### Models
- Pretrained CNN backbone (e.g., MobileNetV3 / ResNet18)
- Transfer learning with ImageNet weights
- Optimized for fast CPU inference

### Training Details
- Image size: 224×224
- Lightweight augmentations
- Early stopping and threshold tuning
- Evaluation with F1-score, precision, recall, and confusion matrices

---

## 📈 Results


---

## 🚀 API Usage

### Endpoint
POST /detect-damage

### Request
- `multipart/form-data`
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

---

🛠️ Running Locally

1. Install dependencies
pip install -r requirements.txt

2. Download datasets
Follow instructions in:
scripts/download_data.md

3. Generate dataset splits
python scripts/make_splits.py

4. Train models
python scripts/train_binary.py
python scripts/train_type.py

5. Start API
uvicorn app.main:app --reload

---

📁 Repository Structure
car-damage-detection-service/
├── app/            # FastAPI service
├── data/           # Dataset splits (raw data excluded)
├── models/         # Trained model artifacts
├── scripts/        # Training and preprocessing scripts
├── tests/          # Unit and API tests
├── Dockerfile
└── README.md

---

⚠️ Limitations & Future Work

Performance may degrade on extreme lighting or occlusions
Domain shift between datasets

Future improvements:
Damage localization (bounding boxes)
Segmentation-based severity estimation
Model quantization for faster CPU inference