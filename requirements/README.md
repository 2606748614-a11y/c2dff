# C2DFF-Net Environment Requirements

This project has been verified in the `c2` environment with GPU enabled:

- Python 3.10
- CUDA available through PyTorch
- GPU tested: NVIDIA GeForce RTX 4090
- Local Ultralytics source package from this repository

Install PyTorch with the CUDA wheel first, then install the common Python
dependencies:

```bash
python -m pip install -r requirements/torch-cu121.txt
python -m pip install -r requirements/runtime.txt
```

If this project is run inside the prebuilt c2 container used for testing here,
the GPU stack may already exist in `/usr/local/lib/python3.10/dist-packages`.
The current conda environment was made able to see those packages through:

```text
/root/.conda/envs/c2/lib/python3.10/site-packages/system-dist-packages.pth
```

Verify the GPU environment:

```bash
python - <<'PY'
import cv2
import torch

print("cv2", cv2.__version__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

