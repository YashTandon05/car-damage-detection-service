import json
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class DamageModelBundle:
    model: torch.jit.ScriptModule
    threshold: float
    image_size: int
    label_to_idx: dict

@dataclass
class TypeModelBundle:
    model: torch.jit.ScriptModule
    classes: list
    image_size: int
    label_to_idx: dict

def load_damage_bundle(model_path: Path, meta_path: Path) -> DamageModelBundle:
    meta = json.loads(meta_path.read_text())
    model = torch.jit.load(str(model_path)).eval()

    threshold = float(meta["threshold"])
    image_size = int(meta["image_size"])
    label_to_idx = meta["label_to_idx"]

    return DamageModelBundle(model=model, threshold=threshold, image_size=image_size, label_to_idx=label_to_idx)

def load_type_bundle(model_path: Path, meta_path: Path) -> TypeModelBundle:
    meta = json.loads(meta_path.read_text())
    model = torch.jit.load(str(model_path)).eval()

    classes = meta["classes"]
    image_size = int(meta["image_size"])
    label_to_idx = meta["label_to_idx"]

    return TypeModelBundle(model=model, classes=classes, image_size=image_size, label_to_idx=label_to_idx)