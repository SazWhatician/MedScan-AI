"""
model_loader.py
Downloads models from HuggingFace repos, caches them, runs inference.
IMPORTANT: Uses exact same architecture + transforms as Colab training code.
"""

import os
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

CACHE_DIR = "/app/model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PNEUMONIA — Keras .h5
# ══════════════════════════════════════════════════════════════════════════════

PNEUMO_REPO   = "SaswatML123/PneuModel"
PNEUMO_FILE   = "pneumodel.h5"
PNEUMO_SIZE   = (224, 224)
_pneumo_model = None


def _download(repo, filename):
    local = os.path.join(CACHE_DIR, filename)
    if os.path.exists(local):
        print(f"[Cache] {filename}")
        return local
    print(f"[HuggingFace] Downloading {filename}...")
    return hf_hub_download(repo_id=repo, filename=filename, local_dir=CACHE_DIR)


def load_pneumo_model():
    global _pneumo_model
    if _pneumo_model is not None:
        return
    import tensorflow as tf
    path = _download(PNEUMO_REPO, PNEUMO_FILE)
    print("[Pneumonia] Loading...")
    _pneumo_model = tf.keras.models.load_model(path)
    print("[Pneumonia] ✓ Ready")


def predict_pneumonia(image: Image.Image) -> dict:
    load_pneumo_model()
    img = image.convert("RGB").resize(PNEUMO_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = _pneumo_model.predict(arr, verbose=0)

    if preds.shape[-1] == 1:
        pneumonia_prob = float(preds[0][0])
        normal_prob    = 1.0 - pneumonia_prob
    else:
        normal_prob    = float(preds[0][0])
        pneumonia_prob = float(preds[0][1])

    if pneumonia_prob >= 0.5:
        label, confidence = "PNEUMONIA", pneumonia_prob
    else:
        label, confidence = "NORMAL", normal_prob

    return {
        "label":         label,
        "confidence":    round(confidence, 4),
        "probabilities": {
            "NORMAL":    round(normal_prob, 4),
            "PNEUMONIA": round(pneumonia_prob, 4),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SKIN CANCER — exact same architecture as Colab training
# ══════════════════════════════════════════════════════════════════════════════

SKIN_REPO  = "SaswatML123/Skin_cancer_detection"
SKIN_FILES = {
    "efficientnetv2m": ("model1_efficientnetv2m.pth", "tf_efficientnetv2_m"),
    "efficientnetv2s": ("model2_efficientnetv2s.pth", "tf_efficientnetv2_s"),
    "convnext":        ("model3_convnext.pth",         "convnext_base"),
}

# Alphabetical sorted order — matches CLASS_NAMES = sorted(df['dx'].unique())
SKIN_CLASSES = [
    "Actinic Keratoses",    # akiec — index 0
    "Basal Cell Carcinoma", # bcc   — index 1
    "Benign Keratosis",     # bkl   — index 2
    "Dermatofibroma",       # df    — index 3
    "Melanoma",             # mel   — index 4
    "Melanocytic Nevi",     # nv    — index 5
    "Vascular Lesions",     # vasc  — index 6
]
NUM_SKIN_CLASSES = len(SKIN_CLASSES)
_skin_models     = []
SKIN_TRANSFORM   = None


# ── Exact replica of Colab SkinCancerModel ────────────────────────────────────
def _build_skin_model(model_name: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm

    class GeM(nn.Module):
        def __init__(self, p=3, eps=1e-6):
            super().__init__()
            self.p   = nn.Parameter(torch.ones(1) * p)
            self.eps = eps

        def forward(self, x):
            return F.avg_pool2d(
                x.clamp(min=self.eps).pow(self.p),
                (x.size(-2), x.size(-1))
            ).pow(1.0 / self.p)

    class SkinCancerModel(nn.Module):
        def __init__(self, num_classes=7, model_name='tf_efficientnetv2_m',
                     pretrained=False, drop_rate=0.3):
            super().__init__()
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained,
                num_classes=0, global_pool='', drop_rate=drop_rate
            )
            in_features = self.backbone.num_features
            self.pool = GeM()
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(drop_rate),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            return self.head(self.pool(self.backbone(x)))

    return SkinCancerModel(
        num_classes=NUM_SKIN_CLASSES,
        model_name=model_name,
        pretrained=False
    )


def load_skin_models():
    global _skin_models, SKIN_TRANSFORM
    if _skin_models:
        return

    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Exact same transforms as Colab get_val_transforms(300)
    _albu_transform = A.Compose([
        A.Resize(height=300, width=300),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Store as a wrapper function
    def transform_fn(pil_img):
        img_np = np.array(pil_img.convert("RGB"))
        return _albu_transform(image=img_np)["image"].unsqueeze(0)

    global SKIN_TRANSFORM
    SKIN_TRANSFORM = transform_fn

    device = torch.device("cpu")

    for arch, (filename, model_name) in SKIN_FILES.items():
        path  = _download(SKIN_REPO, filename)
        model = _build_skin_model(model_name)

        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        _skin_models.append(model)
        print(f"[Skin] ✓ {arch}")

    print(f"[Skin] Ensemble ready — {len(_skin_models)} models")


def predict_skin(image: Image.Image) -> dict:
    import torch
    import torch.nn.functional as F
    load_skin_models()

    img_t = SKIN_TRANSFORM(image)  # (1, 3, 300, 300)

    all_probs = []
    with torch.no_grad():
        for model in _skin_models:
            logits = model(img_t)
            probs  = F.softmax(logits, dim=1).squeeze().numpy()
            all_probs.append(probs)

    avg = np.mean(all_probs, axis=0)
    top = int(np.argmax(avg))

    return {
        "label":         SKIN_CLASSES[top],
        "confidence":    round(float(avg[top]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(SKIN_CLASSES, avg)},
        "model_count":   len(_skin_models),
    }
