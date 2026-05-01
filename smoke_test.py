"""Verify MuseTalk install: imports + CUDA + weight files present."""
import sys
from pathlib import Path

print(f"Python: {sys.version}")

import torch
print(f"torch: {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")

import numpy, transformers, diffusers, mmcv, mmpose, mmdet, mmengine
print(f"numpy {numpy.__version__}  transformers {transformers.__version__}  "
      f"diffusers {diffusers.__version__}")
print(f"mmcv {mmcv.__version__}  mmdet {mmdet.__version__}  "
      f"mmpose {mmpose.__version__}  mmengine {mmengine.__version__}")

sys.path.insert(0, str(Path(__file__).parent))
from musetalk.utils.utils import load_all_model  # noqa: F401
print("musetalk.utils.utils.load_all_model importable")

models = Path(__file__).parent / "models"
required = [
    models / "musetalk" / "pytorch_model.bin",
    models / "musetalk" / "musetalk.json",
    models / "musetalkV15" / "unet.pth",
    models / "musetalkV15" / "musetalk.json",
    models / "sd-vae" / "diffusion_pytorch_model.bin",
    models / "sd-vae" / "config.json",
    models / "whisper" / "pytorch_model.bin",
    models / "whisper" / "config.json",
    models / "whisper" / "preprocessor_config.json",
    models / "dwpose" / "dw-ll_ucoco_384.pth",
    models / "syncnet" / "latentsync_syncnet.pt",
    models / "face-parse-bisent" / "79999_iter.pth",
    models / "face-parse-bisent" / "resnet18-5c106cde.pth",
]
missing = [str(p.relative_to(models.parent)) for p in required if not p.exists()]
if missing:
    print("MISSING WEIGHTS:")
    for p in missing:
        print(f"  - {p}")
    sys.exit(1)
print(f"All {len(required)} weight files present.")
