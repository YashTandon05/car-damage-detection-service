from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = REPO_ROOT / "models"
DAMAGE_MODEL_PATH = MODELS_DIR / "damage_detector.pt"
DAMAGE_META_PATH  = MODELS_DIR / "damage_detector_meta.json"

TYPE_MODEL_PATH = MODELS_DIR / "damage_type.pt"
TYPE_META_PATH  = MODELS_DIR / "damage_type_meta.json"

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB