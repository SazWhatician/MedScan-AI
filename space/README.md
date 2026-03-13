---
title: MediScan AI
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# MediScan AI — Medical Diagnostic API

FastAPI backend serving skin cancer and pneumonia predictions.

## Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Status |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/predict/pneumonia` | Chest X-ray analysis |
| POST | `/predict/skin` | Skin lesion classification |

## Models
- **Pneumonia**: Custom CNN (`SaswatML123/PneuModel`)
- **Skin Cancer**: EfficientNetV2M + EfficientNetV2S + ConvNeXt ensemble (`SaswatML123/Skin_cancer_detection`)

> For research use only. Not a medical device.
