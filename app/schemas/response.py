from typing import List, Optional
from pydantic import BaseModel, Field

class TypePredictionOut(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)

class DetectDamageResponse(BaseModel):
    damage_detected: bool
    damage_confidence: float = Field(ge=0.0, le=1.0)
    damage_type: Optional[str] = None
    type_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_2_types: Optional[List[TypePredictionOut]] = None
    model_version: str
    latency_ms: int