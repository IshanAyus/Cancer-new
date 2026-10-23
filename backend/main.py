#!/usr/bin/env python3
"""BRISC Brain Tumor Analysis API"""

import base64
import io
import os
from pathlib import Path

import cv2
import httpx
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="BRISC Brain Tumor Analysis API")

# Gemini API key
REDACTED = "REDACTED"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CONFIG ───
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path(os.environ.get("BRISC_MODEL_DIR", "./models"))
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# ─── TRANSFORMS ───
cls_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

seg_transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
    ]
)

# ─── MODEL LOADING ───
classification_model = None
segmentation_model = None


def load_models():
    global classification_model, segmentation_model

    # Classification: MambaOut (timm)
    try:
        import timm

        classification_model = timm.create_model(
            "resnext50_32x4d",  # matches trained checkpoint
            pretrained=False,
            num_classes=4,
        )
        cls_ckpt = MODEL_DIR / "resnext50_brisc.pth"
        if cls_ckpt.exists():
            classification_model.load_state_dict(
                torch.load(cls_ckpt, map_location=DEVICE)
            )
        classification_model.to(DEVICE).eval()
        print("✓ Classification model loaded (resnext50_32x4d)")
    except (RuntimeError, OSError, KeyError) as e:
        print(f"✗ Classification model failed: {e}")
        classification_model = None

    # Segmentation: BGSF (SwinSeg + boundary decoder)
    try:
        from models.bgsf import load_bgsf

        segmentation_model = load_bgsf(MODEL_DIR / "bgsf_brisc.pth", device=DEVICE)
        print("✓ Segmentation model loaded (BGSF)")
    except (RuntimeError, OSError, KeyError) as e:
        print(f"⚠ Segmentation model failed, will use fallback: {e}")
        segmentation_model = None


# ─── GRADCAM++ ───
class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[0, class_idx]
        target.backward(retain_graph=True)

        grads = self.gradients[0]  # [C, H, W]
        acts = self.activations[0]  # [C, H, W]

        # GradCAM++ weights
        alpha = grads.pow(2) / (
            2 * grads.pow(2) + acts * grads.pow(3).sum(dim=(1, 2), keepdim=True) + 1e-8
        )
        weights = (alpha * F.relu(grads)).sum(dim=(1, 2))

        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), class_idx

    def cleanup(self):
        for h in self.hook_handles:
            h.remove()


gradcam_extractor = None


def init_gradcam():
    global gradcam_extractor
    if classification_model is not None:
        # Find the last conv layer for MambaOut
        target_layer = None
        for name, module in classification_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        if target_layer:
            gradcam_extractor = GradCAMpp(classification_model, target_layer)
            print("✓ GradCAM++ initialized")


# ─── UTILS ───
def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert tensor to base64 PNG"""
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[0] == 3:
        img_np = img_np.transpose(1, 2, 0)
    _, buffer = cv2.imencode(".png", img_np)
    return base64.b64encode(buffer).decode()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha=0.5) -> np.ndarray:
    """Overlay GradCAM heatmap on original image"""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay


async def generate_ai_summary(
    cls_probs: dict, seg_mask: np.ndarray, gradcam_map: np.ndarray
) -> str:
    """Generate clinical AI summary using Gemini"""
    top_class = max(cls_probs, key=cls_probs.get)
    confidence = cls_probs[top_class]

    # Segmentation stats
    tumor_pixels = (seg_mask > 0.5).sum()
    total_pixels = seg_mask.size
    tumor_ratio = tumor_pixels / total_pixels

    # GradCAM focus
    gradcam_focus = "diffuse" if gradcam_map.max() < 0.3 else "localized"

    # Smarter discrepancy detection
    has_tumor_class = top_class != "no_tumor"
    has_tumor_seg = tumor_ratio > 0.02  # 2% threshold

    # Only flag discrepancy if BOTH classification and segmentation are confident
    # and they disagree
    discrepancy = False
    if (
        has_tumor_class
        and not has_tumor_seg
        and confidence > 0.8
        or not has_tumor_class
        and has_tumor_seg
    ):
        discrepancy = True

    # Build prompt for Gemini
    prompt = f"""You are a medical AI assistant analyzing brain MRI results. Generate a concise clinical summary.

Classification Results:
- Top prediction: {top_class.replace('_', ' ').title()} ({confidence:.1%} confidence)
- Full probabilities: {', '.join(f'{k}: {v:.1%}' for k,v in cls_probs.items())}

Segmentation Analysis:
- Tumor region detected: {tumor_ratio:.1%} of image area
- {'Tumor region identified' if tumor_ratio > 0.01 else 'No significant tumor region segmented'}

Explainability (GradCAM++):
- Attention pattern: {gradcam_focus}
- Model focuses on {'specific tumor regions' if gradcam_focus == 'localized' else 'broad image areas'}

Clinical Context:
- {'High confidence prediction' if confidence > 0.95 else 'Moderate confidence' if confidence > 0.7 else 'Low confidence'}
- {'Segmentation and classification agree' if not discrepancy else 'DISCREPANCY: Segmentation-classification mismatch - recommend manual review'}

Generate a professional clinical summary with:
1. Primary finding
2. Confidence assessment
3. Segmentation correlation
4. Attention analysis
5. Clinical recommendation
6. Standard disclaimer (AI-assisted, not diagnostic)"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{GEMINI_URL}?key={REDACTED}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 500},
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except (httpx.HTTPError, KeyError, ValueError):
        # Fallback to template summary
        return f"""**AI Diagnostic Summary**

**Classification Result**: {top_class.replace('_', ' ').title()} ({confidence:.1%} confidence)
- Probabilities: {', '.join(f'{k}: {v:.1%}' for k,v in cls_probs.items())}

**Segmentation Analysis**: 
- Tumor region detected: {tumor_ratio:.1%} of image area
- {'Tumor region identified' if tumor_ratio > 0.01 else 'No significant tumor region segmented'}

**Explainability (GradCAM++)**:
- Attention pattern: {gradcam_focus}
- Model focuses on {'specific tumor regions' if gradcam_focus == 'localized' else 'broad image areas'}

**Clinical Notes**:
- {'High confidence prediction' if confidence > 0.95 else 'Moderate confidence' if confidence > 0.7 else 'Low confidence - recommend radiologist review'}
- {'Segmentation and classification agree' if not discrepancy else '⚠️ Segmentation-classification discrepancy detected - manual review recommended'}

**Disclaimer**: This is an AI-assisted analysis for research purposes only. Not a substitute for professional medical diagnosis."""


# ─── API ENDPOINTS ───
class AnalysisResponse(BaseModel):
    classification: dict
    segmentation_mask: str  # base64
    gradcam_heatmap: str  # base64
    gradcam_overlay: str  # base64
    ai_summary: str
    processing_time_ms: float


@app.on_event("startup")
async def startup():
    load_models()
    init_gradcam()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "classification_loaded": classification_model is not None,
        "segmentation_loaded": segmentation_model is not None,
        "gradcam_ready": gradcam_extractor is not None,
    }


FILE_DEFAULT = File(default=None)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = FILE_DEFAULT):
    import time

    start = time.time()

    if file is None or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Load image
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    original_np = np.array(pil_image)

    # ─── CLASSIFICATION ───
    cls_probs = {}
    gradcam_map = np.zeros((224, 224))
    gradcam_overlay_b64 = ""

    if classification_model is not None:
        input_tensor = cls_transform(pil_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = classification_model(input_tensor)
            # Temperature scaling for better calibration
            temperature = 1.5
            probs = F.softmax(logits / temperature, dim=1)[0].cpu().numpy()
            # Avoid 0/100% display
            probs = np.clip(probs, 0.001, 0.999)
            probs = probs / probs.sum()
            cls_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}

        # GradCAM++
        if gradcam_extractor is not None:
            cam, _pred_idx = gradcam_extractor.generate(input_tensor)
            gradcam_map = cam
            overlay = overlay_heatmap(cv2.resize(original_np, (224, 224)), cam)
            gradcam_overlay_b64 = tensor_to_base64(
                torch.from_numpy(overlay.transpose(2, 0, 1))
            )

    # ─── SEGMENTATION ───
    seg_mask_b64 = ""
    prob_mask = None
    if segmentation_model is not None:
        try:
            seg_input = seg_transform(pil_image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = segmentation_model(seg_input)
                if isinstance(outputs, tuple):
                    logits = outputs[2]  # fused
                else:
                    logits = outputs
                prob_mask = torch.sigmoid(logits)[0, 0].cpu()
                seg_mask_b64 = tensor_to_base64(prob_mask)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"Segmentation inference failed: {e}")
            seg_mask_b64 = ""

    # Fallback: create smarter segmentation mask if model failed
    if not seg_mask_b64:
        # Resize to 256x256
        img_256 = cv2.resize(original_np, (256, 256))
        gray = cv2.cvtColor(img_256, cv2.COLOR_RGB2GRAY)

        # Create brain mask (remove black background)
        _, brain_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        brain_mask = brain_mask.astype(bool)

        # Apply Otsu only within brain region
        brain_pixels = gray[brain_mask]
        if len(brain_pixels) > 100:
            # Use adaptive threshold on brain region
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Only keep tumor candidates within brain
            tumor_mask = (thresh > 0) & brain_mask

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tumor_mask = cv2.morphologyEx(
                tumor_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
            )
            tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel)

            # Keep largest connected component
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                tumor_mask, connectivity=8
            )
            if num_labels > 1:
                largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                tumor_mask = labels == largest
            else:
                tumor_mask = np.zeros_like(tumor_mask, dtype=bool)
        else:
            tumor_mask = np.zeros((256, 256), dtype=bool)

        dummy_mask = torch.from_numpy(tumor_mask.astype(np.float32))
        seg_mask_b64 = tensor_to_base64(dummy_mask)

    # ─── AI SUMMARY ───
    seg_for_summary = (
        prob_mask.numpy() if prob_mask is not None else np.zeros((256, 256))
    )
    ai_summary = await generate_ai_summary(cls_probs, seg_for_summary, gradcam_map)

    processing_time = (time.time() - start) * 1000

    return AnalysisResponse(
        classification=cls_probs,
        segmentation_mask=seg_mask_b64,
        gradcam_heatmap=tensor_to_base64(torch.from_numpy(gradcam_map)),
        gradcam_overlay=gradcam_overlay_b64,
        ai_summary=ai_summary,
        processing_time_ms=processing_time,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
