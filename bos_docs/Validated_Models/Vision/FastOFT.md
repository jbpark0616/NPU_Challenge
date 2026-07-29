---
title: Untitled
source: https://developers.bos-semi.com/docs/Validated_Models/Vision/FastOFT/
---

# FastOFt

OFT (Orthographic Feature Transform) Model

OFT is a 3D object detection model that uses orthographic feature transforms to detect objects in 3D space. The model combines a ResNet-based frontend with specialized orthographic feature transformation layers and a topdown refinement network.

## Model Architecture[](#model-architecture)

The OFT model consists of several key components:

- **Frontend**: ResNet-18/34 backbone for feature extraction at multiple scales (8x, 16x, 32x downsampling)
- **Lateral Layers**: Convert ResNet outputs to a common 256-channel feature representation
- **OFT Layers**: Orthographic Feature Transform modules that project features into bird's-eye view
- **Topdown Network**: 8-layer refinement network using BasicBlock modules
- **Detection Head**: Final convolutional layer that outputs object scores, positions, dimensions, and angles
- **Decoder**Additional module that is used to decode encoded outputs into objects

The model outputs:

- **Scores**: Object detection confidence scores
- **Position Offsets**: 3D position predictions (x, y, z)
- **Dimension Offsets**: Object size predictions (width, height, length)
- **Angle Offsets**: Object orientation predictions (sin, cos components)
- **Objects**: Decoded outputs into list of detected objects.

## Project Structure[](#project-structure)

```
models/bos_model/oft/
├── demo/              # Demo scripts and visualization
├── reference/         # PyTorch reference implementation
├── resources/         # Test images and calibration files
├── tests/             # All tests together
    └── pcc/           # Unit tests for individual components
└── tt/                # TenstorrentNN (TTNN) optimized implementation
```
**Input Requirements:**
Both demos require:

- env variable CHECKPOINTS_PATH with pre-trained checkpoint file (e.g., `export CHECKPOINTS_PATH="*your-path*/checkpoint-0600.pth"`)
- Input images in JPG format (located in `resources/`)
- Corresponding calibration files in TXT format (camera intrinsic parameters)

### demo.py[](#demopy)

Full end-to-end inference demo that runs both PyTorch reference and TTNN implementations, comparing their outputs and generating visualizations.

**Features:**

- Loads pre-trained model weights from checkpoint
- Processes input images with calibration data
- Runs full OFT inference pipeline on both CPU (PyTorch) and device (TTNN)
- **Executes complete pipeline on TTNN**: OFTNet model inference + object decoder/encoder
- Compares intermediate outputs and final predictions
- Generates detection visualizations and heatmaps
- Supports various precision modes (float32, bfloat16)
- Configurable fallback modes for debugging

**Usage:**

```
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/bos_model/oft/demo/demo.py
```
### host_demo.py[](#host_demopy)

Host-only demo that compares float32 and bfloat16 precision using only PyTorch reference implementation.

**Features:**

- Precision comparison between fp32 and bfp16
- Object detection visualization
- Performance and accuracy analysis
- No device execution required - pure CPU inference
- Useful for baseline validation

**Usage:**

```
pytest models/bos_model/oft/demo/host_demo.py
```

## 이미지

![BOS_Logo_4E_RGB.avif](../../_assets/BOS_Logo_4E_RGB.avif)  <!-- https://developers.bos-semi.com/img/BOS_Logo_4E_RGB.avif -->
