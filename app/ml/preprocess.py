from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_eval_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_image_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(BytesIO(data))
    return im.convert("RGB")

def preprocess_pil(im: Image.Image, image_size: int) -> torch.Tensor:
    tf = build_eval_transform(image_size)
    x = tf(im).unsqueeze(0)  # (1,3,H,W)
    return x