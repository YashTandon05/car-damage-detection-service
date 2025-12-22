from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

import torch

from app.core.errors import ModelNotLoadedError
from app.ml.model_loader import DamageModelBundle, TypeModelBundle
from app.ml.preprocess import preprocess_pil

@dataclass
class TypePrediction:
    label: str
    confidence: float

@dataclass
class DamageResponseInternal:
    damage_detected: bool
    damage_confidence: float
    damage_type: Optional[str]
    type_confidence: Optional[float]
    top_2_types: Optional[List[TypePrediction]]
    latency_ms: int

class Predictor:
    def __init__(self, damage: DamageModelBundle, dtype: TypeModelBundle):
        self.damage = damage
        self.dtype = dtype

        # Expect mapping: {"no_damage":0, "damage":1}
        self.damage_idx = int(self.damage.label_to_idx["damage"])

    @torch.no_grad()
    def predict(self, pil_image) -> DamageResponseInternal:
        if self.damage is None or self.dtype is None:
            raise ModelNotLoadedError("Models not loaded.")

        t0 = time.perf_counter()

        # ---- Binary
        x_bin = preprocess_pil(pil_image, self.damage.image_size)
        logits = self.damage.model(x_bin)
        probs = torch.softmax(logits, dim=1)[0]
        p_damage = float(probs[self.damage_idx].item())

        damage_detected = p_damage >= self.damage.threshold

        # ---- If no damage: short-circuit
        if not damage_detected:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return DamageResponseInternal(
                damage_detected=False,
                damage_confidence=p_damage,
                damage_type=None,
                type_confidence=None,
                top_2_types=None,
                latency_ms=latency_ms,
            )

        # ---- Type classification
        x_type = preprocess_pil(pil_image, self.dtype.image_size)
        tlogits = self.dtype.model(x_type)
        tprobs = torch.softmax(tlogits, dim=1)[0]  # (num_classes,)

        # top-2
        topk = torch.topk(tprobs, k=min(2, tprobs.numel()))
        top2 = []
        for idx, conf in zip(topk.indices.tolist(), topk.values.tolist()):
            top2.append(TypePrediction(label=self.dtype.classes[int(idx)], confidence=float(conf)))

        latency_ms = int((time.perf_counter() - t0) * 1000)

        return DamageResponseInternal(
            damage_detected=True,
            damage_confidence=p_damage,
            damage_type=top2[0].label,
            type_confidence=top2[0].confidence,
            top_2_types=top2,
            latency_ms=latency_ms,
        )