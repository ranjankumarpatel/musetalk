"""
MuseTalk sidecar HTTP service.

Runs inside MuseTalk's own virtualenv (which ships torch+cu118, mmpose, diffusers,
face-alignment, etc.) and exposes a small JSON API that the local-live-avatar
FastAPI server can call over localhost.

Start:
    cd c:/github/MuseTalk
    .venv/Scripts/python.exe musetalk_service.py --host 127.0.0.1 --port 8001

Endpoints
---------
GET  /health
    returns readiness, device, version, cache size

POST /portrait  (multipart/form-data, file field "image")
    returns {portrait_id, width, height, bbox}
    Runs face detection + face-parse + VAE encode once per portrait and
    caches the result in-memory. Re-uploading the same image returns a
    fresh portrait_id (no de-dupe -- trivial caller concern).

POST /infer  (application/json)
    Body: {portrait_id, pcm_b64, sample_rate, fps=25, jpeg_quality=82}
        pcm_b64 = base64 of raw float32 little-endian PCM (mono).
    returns {frames: [b64 jpeg, ...], count, duration_s, elapsed_ms}

Notes
-----
* All inference runs fp16 on cuda:0 (or cpu if CUDA unavailable).
* Batched over N frames per call; call this per TTS sentence from the caller.
* Single-process, no session isolation beyond the portrait cache -- the caller
  must serialize calls if it opens multiple sessions on 6GB VRAM.
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# MuseTalk's preprocessing.py prints CJK glyphs via plain `print(...)`. On a
# Windows console with cp1252, this raises UnicodeEncodeError mid-request.
# Force UTF-8 on both streams before importing anything that might print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import cv2
import librosa
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from transformers import WhisperModel

_REPO_ROOT = Path(__file__).resolve().parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

from musetalk.utils.utils import load_all_model  # noqa: E402
from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402
from musetalk.utils.preprocessing import get_landmark_and_bbox  # noqa: E402
from musetalk.utils.blending import get_image_prepare_material, get_image_blending  # noqa: E402
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402

logger = logging.getLogger("musetalk_service")

VERSION = "v15"
UNET_CONFIG = "./models/musetalkV15/musetalk.json" if VERSION == "v15" else "./models/musetalk/musetalk.json"
UNET_WEIGHTS = "./models/musetalkV15/unet.pth" if VERSION == "v15" else "./models/musetalk/pytorch_model.bin"
VAE_TYPE = "sd-vae"
WHISPER_DIR = "./models/whisper"
PARSING_MODE = "jaw"
EXTRA_MARGIN = 10
LEFT_CHEEK_WIDTH = 90
RIGHT_CHEEK_WIDTH = 90
BBOX_SHIFT = 0

DEVICE: torch.device
WEIGHT_DTYPE: torch.dtype
VAE = None
UNET = None
PE = None
WHISPER = None
AUDIO_PROCESSOR: Optional[AudioProcessor] = None
FACE_PARSER = None
TIMESTEPS: Optional[torch.Tensor] = None

PORTRAIT_CACHE: dict[str, dict] = {}


class InferBody(BaseModel):
    portrait_id: str
    pcm_b64: str = Field(..., description="base64 of raw float32 LE mono PCM")
    sample_rate: int
    fps: int = 25
    jpeg_quality: int = 82


class InferResponse(BaseModel):
    frames: list[str]
    count: int
    duration_s: float
    elapsed_ms: float


app = FastAPI(title="MuseTalk Sidecar", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    global DEVICE, WEIGHT_DTYPE, VAE, UNET, PE, WHISPER, AUDIO_PROCESSOR, FACE_PARSER, TIMESTEPS

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("musetalk_service: device=%s", DEVICE)

    VAE, UNET, PE = load_all_model(
        unet_model_path=UNET_WEIGHTS,
        vae_type=VAE_TYPE,
        unet_config=UNET_CONFIG,
        device=DEVICE,
    )
    TIMESTEPS = torch.tensor([0], device=DEVICE)

    PE = PE.half().to(DEVICE)
    VAE.vae = VAE.vae.half().to(DEVICE)
    UNET.model = UNET.model.half().to(DEVICE)
    WEIGHT_DTYPE = UNET.model.dtype

    AUDIO_PROCESSOR = AudioProcessor(feature_extractor_path=WHISPER_DIR)
    whisper = WhisperModel.from_pretrained(WHISPER_DIR)
    whisper = whisper.to(device=DEVICE, dtype=WEIGHT_DTYPE)
    whisper.train(False)  # inference mode (equivalent to Module.eval())
    whisper.requires_grad_(False)
    WHISPER = whisper

    if VERSION == "v15":
        FACE_PARSER = FaceParsing(
            left_cheek_width=LEFT_CHEEK_WIDTH,
            right_cheek_width=RIGHT_CHEEK_WIDTH,
        )
    else:
        FACE_PARSER = FaceParsing()

    logger.info("musetalk_service: ready")


@app.get("/health")
def health() -> dict:
    return {
        "ready": VAE is not None and UNET is not None,
        "device": str(DEVICE) if "DEVICE" in globals() else "unknown",
        "version": VERSION,
        "portraits_cached": len(PORTRAIT_CACHE),
    }


def _prepare_portrait(image_bgr: np.ndarray) -> dict:
    tmp_path = _REPO_ROOT / "results" / f"_tmp_{uuid.uuid4().hex}.png"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp_path), image_bgr)

    try:
        coord_list, frame_list = get_landmark_and_bbox([str(tmp_path)], BBOX_SHIFT)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    if not coord_list or coord_list[0] == (0.0, 0.0, 0.0, 0.0):
        raise HTTPException(status_code=422, detail="face not detected in portrait")

    bbox = coord_list[0]
    frame_bgr = frame_list[0]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if VERSION == "v15":
        y2 = min(y2 + EXTRA_MARGIN, frame_bgr.shape[0])
        bbox = (x1, y1, x2, y2)

    crop = frame_bgr[y1:y2, x1:x2]
    crop_256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    latent = VAE.get_latents_for_unet(crop_256)

    mode = PARSING_MODE if VERSION == "v15" else "raw"
    mask, mask_crop_box = get_image_prepare_material(
        frame_bgr, list(bbox), fp=FACE_PARSER, mode=mode
    )

    return {
        "frame_bgr": frame_bgr,
        "bbox": bbox,
        "latent": latent,
        "mask": mask,
        "mask_crop_box": mask_crop_box,
    }


@app.post("/portrait")
async def upload_portrait(image: UploadFile = File(...)) -> dict:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc}") from exc

    frame_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    entry = _prepare_portrait(frame_bgr)
    portrait_id = uuid.uuid4().hex
    PORTRAIT_CACHE[portrait_id] = entry

    h, w = entry["frame_bgr"].shape[:2]
    return {
        "portrait_id": portrait_id,
        "width": w,
        "height": h,
        "bbox": list(entry["bbox"]),
    }


@app.delete("/portrait/{portrait_id}")
def drop_portrait(portrait_id: str) -> dict:
    existed = PORTRAIT_CACHE.pop(portrait_id, None) is not None
    return {"dropped": existed}


def _whisper_chunks_from_pcm(
    pcm_f32_16k: np.ndarray,
    fps: int,
) -> torch.Tensor:
    assert AUDIO_PROCESSOR is not None
    assert WHISPER is not None

    sr = 16000
    segment_length = 30 * sr
    segments = [
        pcm_f32_16k[i : i + segment_length]
        for i in range(0, len(pcm_f32_16k), segment_length)
    ]

    features = []
    for seg in segments:
        feat = AUDIO_PROCESSOR.feature_extractor(
            seg, return_tensors="pt", sampling_rate=sr
        ).input_features
        feat = feat.to(dtype=WEIGHT_DTYPE)
        features.append(feat)

    chunks = AUDIO_PROCESSOR.get_whisper_chunk(
        features,
        DEVICE,
        WEIGHT_DTYPE,
        WHISPER,
        len(pcm_f32_16k),
        fps=fps,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    )
    return chunks


@torch.no_grad()
def _infer_frames(
    entry: dict,
    pcm_f32_16k: np.ndarray,
    fps: int,
    batch_size: int = 8,
) -> list[np.ndarray]:
    whisper_chunks = _whisper_chunks_from_pcm(pcm_f32_16k, fps)
    num_frames = whisper_chunks.shape[0]
    if num_frames == 0:
        return []

    frame_bgr = entry["frame_bgr"]
    bbox = entry["bbox"]
    latent = entry["latent"]
    mask = entry["mask"]
    mask_crop_box = entry["mask_crop_box"]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    out_frames: list[np.ndarray] = []
    for start in range(0, num_frames, batch_size):
        batch = whisper_chunks[start : start + batch_size].to(
            device=DEVICE, dtype=WEIGHT_DTYPE
        )
        audio_feat = PE(batch)
        latent_batch = latent.to(device=DEVICE, dtype=UNET.model.dtype)
        latent_batch = latent_batch.expand(batch.shape[0], -1, -1, -1)

        pred_latents = UNET.model(
            latent_batch,
            TIMESTEPS,
            encoder_hidden_states=audio_feat,
        ).sample
        pred_latents = pred_latents.to(device=DEVICE, dtype=VAE.vae.dtype)
        recon = VAE.decode_latents(pred_latents)

        for res_face in recon:
            face = cv2.resize(res_face.astype(np.uint8), (x2 - x1, y2 - y1))
            combined = get_image_blending(
                frame_bgr.copy(), face, list(bbox), mask, mask_crop_box
            )
            out_frames.append(combined)

    return out_frames


@app.post("/infer", response_model=InferResponse)
def infer(body: InferBody) -> InferResponse:
    entry = PORTRAIT_CACHE.get(body.portrait_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="portrait_id not found")

    try:
        pcm_bytes = base64.b64decode(body.pcm_b64)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad pcm_b64: {exc}") from exc

    pcm = np.frombuffer(pcm_bytes, dtype=np.float32).copy()
    if pcm.size == 0:
        return InferResponse(frames=[], count=0, duration_s=0.0, elapsed_ms=0.0)

    if body.sample_rate != 16000:
        pcm_16k = librosa.resample(pcm, orig_sr=body.sample_rate, target_sr=16000)
    else:
        pcm_16k = pcm
    pcm_16k = pcm_16k.astype(np.float32, copy=False)
    duration_s = float(len(pcm_16k) / 16000.0)

    t0 = time.monotonic()
    frames_bgr = _infer_frames(entry, pcm_16k, body.fps)

    encoded: list[str] = []
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(body.jpeg_quality)]
    for frame in frames_bgr:
        ok, buf = cv2.imencode(".jpg", frame, jpeg_params)
        if not ok:
            raise HTTPException(status_code=500, detail="jpeg encode failed")
        encoded.append(base64.b64encode(buf.tobytes()).decode("ascii"))

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    return InferResponse(
        frames=encoded,
        count=len(encoded),
        duration_s=duration_s,
        elapsed_ms=elapsed_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8001, type=int)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
