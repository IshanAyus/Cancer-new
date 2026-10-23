# BRISC Brain Tumor Analysis Web App

A complete web application for brain tumor classification, segmentation, and explainability using your trained BRISC models.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   React/Vite    │───▶│    FastAPI      │
│   Frontend      │     │    Backend      │
│   (Port 3000)   │     │    (Port 8000)  │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │ MambaOut │ │   BGSF   │ │ GradCAM++│
             │ Classify │ │ Segment  │ │ Explain  │
             └──────────┘ └──────────┘ └──────────┘
```

## Quick Start

```bash
cd webapp

# 1. Add your model checkpoints
cp /path/to/mambaout_brisc.pth backend/models/
cp /path/to/bgsf_brisc.pth backend/models/

# 2. Run everything
./start.sh
```

Then open **http://localhost:3000**

## Model Checkpoints Required

Place these in `backend/models/`:

| File | Description |
|------|-------------|
| `mambaout_brisc.pth` | MambaOut classification (4 classes) |
| `bgsf_brisc.pth` | BGSF segmentation (Swin-Tiny + boundary decoder) |

**Classification classes**: `glioma`, `meningioma`, `no_tumor`, `pituitary`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service status |
| `POST` | `/analyze` | Upload MRI → returns classification, segmentation, GradCAM++, AI summary |

### `/analyze` Response

```json
{
  "classification": {"glioma": 0.01, "meningioma": 0.98, "no_tumor": 0.005, "pituitary": 0.005},
  "segmentation_mask": "base64_png",
  "gradcam_heatmap": "base64_png", 
  "gradcam_overlay": "base64_png",
  "ai_summary": "Clinical text summary...",
  "processing_time_ms": 245
}
```

## Features

1. **Drag & Drop Upload** - MRI images (PNG, JPG)
2. **Classification** - 4-class tumor type with probabilities
3. **Segmentation** - BGSF tumor mask (probability map)
4. **GradCAM++ Explainability** - Attention heatmap + overlay
5. **AI Clinical Summary** - Auto-generated report with:
   - Top prediction + confidence
   - Segmentation agreement check
   - Attention pattern analysis
   - Discrepancy warnings

## Project Structure

```
webapp/
├── start.sh                 # One-command startup
├── backend/
│   ├── main.py              # FastAPI app
│   ├── requirements.txt     # Python deps
│   └── models/
│       ├── __init__.py
│       └── bgsf.py          # BGSF architecture
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx          # Main UI
        └── styles.css
```

## Customization

### Change Model Architecture
Edit `backend/models/bgsf.py` - matches your paper's BGSF exactly.

### Modify GradCAM Target Layer
In `main.py`, adjust the layer detection in `init_gradcam()`:
```python
# For MambaOut, find the last conv layer
for name, module in classification_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        target_layer = module  # Last one wins
```

### Add More Explainability
The `generate_ai_summary()` function in `main.py` creates the clinical report - customize the logic there.

## Production Deployment

```bash
# Backend
cd backend && gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Frontend
cd frontend && npm run build
# Serve dist/ with nginx
```

## GPU Support

The backend auto-detects CUDA. For production:
```bash
docker run --gpus all -p 8000:8000 -v ./models:/app/models brisc-backend
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Model not found" | Check `backend/models/` has `.pth` files |
| CUDA OOM | Reduce batch size or use CPU: `DEVICE=cpu` |
| CORS errors | Frontend proxy in `vite.config.js` handles this |
| Slow inference | Enable `torch.compile()` in `main.py` |

## License

Research use only. Not for clinical deployment without validation.
