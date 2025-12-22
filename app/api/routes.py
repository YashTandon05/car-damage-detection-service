from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from PIL import UnidentifiedImageError

from app.core.config import MAX_IMAGE_BYTES
from app.core.errors import ModelNotLoadedError
from app.ml.preprocess import load_image_from_bytes
from app.schemas.response import DetectDamageResponse, TypePredictionOut

router = APIRouter()


@router.get("/health")
def health(request: Request):
    pred = getattr(request.app.state, "predictor", None)
    if pred is None:
        raise HTTPException(status_code=500, detail="Predictor not loaded.")

    return {
        "status": "ok",
        "binary_threshold": pred.damage.threshold,
        "binary_image_size": pred.damage.image_size,
        "type_image_size": pred.dtype.image_size,
        "type_classes": pred.dtype.classes,
    }


@router.post("/detect-damage", response_model=DetectDamageResponse)
async def detect_damage(request: Request, file: UploadFile = File(...)):
    pred = getattr(request.app.state, "predictor", None)
    if pred is None:
        raise HTTPException(status_code=500, detail="Predictor not loaded.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 5MB).")

    try:
        pil = load_image_from_bytes(data)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image. The file may be corrupted.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        out = pred.predict(pil)
    except ModelNotLoadedError:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return DetectDamageResponse(
        damage_detected=out.damage_detected,
        damage_confidence=out.damage_confidence,
        damage_type=out.damage_type,
        type_confidence=out.type_confidence,
        top_2_types=None if out.top_2_types is None else [
            TypePredictionOut(label=t.label, confidence=t.confidence) for t in out.top_2_types
        ],
        model_version="1.0.0",
        latency_ms=out.latency_ms,
    )