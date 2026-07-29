---
title: Untitled
source: https://developers.bos-semi.com/docs/Validated_Models/Vision/PDL/
---

# PDL

Panoptic DeepLab

## Platforms:[](#platforms)

Made for BOS chips, mostly tested on Blackhole with a core grid of 20 cores.

## Introduction[](#introduction)

Panoptic DeepLab is a unified model for panoptic segmentation that combines semantic segmentation and instance segmentation into a single framework. The model uses a shared ResNet backbone with separate heads for semantic segmentation and instance embedding prediction, enabling comprehensive scene understanding by simultaneously identifying both "stuff" (background regions like road, sky) and "things" (countable objects like cars, people).

## Prerequisites[](#prerequisites)

- Clone the [tt-metal](https://github.com/tenstorrent/tt-metal)repository to access the source code
- Install [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Setup[](#setup)

### Download Weights[](#download-weights)

Download the Panoptic-DeepLab CityScapes weights file:
[[https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/model_final_bd324a.pkl](https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/model_final_bd324a.pkl)]

Place the downloaded `model_final_bd324a.pkl` file in `models/bos_model/panoptic_deeplab/weights/`

## How to Run[](#how-to-run)

### Run the Full Model Test[](#run-the-full-model-test)

```
# From tt-metal root directory
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_tt_model.py
```
### Run Component Tests[](#run-component-tests)

```
# Test ASPP component
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_aspp.py
# Test ResNet backbone
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_resnet.py
# Test semantic segmentation head
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_semseg.py
# Test instance embedding head
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_insemb.py
```
### Run Device Performance Tests[](#run-device-performance-tests)

```
# Test full model performance
pytest models/bos_model/panoptic_deeplab/tests/test_pipeline_e2e.py
```
### Run the Demo[](#run-the-demo)

```
# Single image with custom output directory
python models/bos_model/panoptic_deeplab/run_panoptic_deeplab.py image_or_directory_path weights_path output_dir
```
Use the following arguments:

- `--trace`enables running in Trace mode.
- `-p`enables Persistent cache
- `--enable_logger`enables Logger output
- `--model_category`selects the model variant:- `DEEPLAB_V3_PLUS`: (default): Semantic segmentation only. It's faster because it doesn't compute center and offset heads, which the full panoptic model uses for instance segmentation
- `PANOPTIC_DEEPLAB`: Full model with semantic and instance segmentation (center and offset heads)
 

### Demo Output Files[](#demo-output-files)

The demo generates several output files for each processed image in separate directories:

**TTNN Output** (`ttnn_output/`):

- `{image_name}_original.jpg`: Original input image
- `{image_name}_panoptic.jpg`: TTNN panoptic segmentation visualization (blended with original)

**PyTorch Reference** (`pytorch_output/`):

- `{image_name}_original.jpg`: Original input image
- `{image_name}_panoptic.jpg`: PyTorch reference panoptic segmentation visualization

### Image Requirements[](#image-requirements)

- **Best Results**: Street scene images similar to Cityscapes dataset
- **Supported Formats**: jpg, jpeg, png, bmp, tiff
- **Auto-Resize**: All images are automatically resized to 512x1024 for inference
- **Classes**: Predicts 19 Cityscapes classes (road, car, person, etc.)

## Details[](#details)

- The entry point to the TTNN Panoptic DeepLab model is `TtPanopticDeepLab`in`models/experimental/panoptic_deeplab/tt/tt_model.py`. The model uses weights from the Detectron2 implementation.

**Input Size: 512x1024**

- Input size is optimized for Cityscapes dataset format with height=512 and width=1024.

**Batch Size: 1**

- Current implementation uses batch size of 1 for optimal memory usage with L1 memory configuration.

**Memory Configuration**

- The model is currently configured to run primarily in DRAM. With further optimization, execution will shift predominantly to L1 memory.

### Model Components[](#model-components)

- **ResNet Backbone**: Provides hierarchical feature extraction with multiple resolution levels
- **ASPP Module**: Atrous Spatial Pyramid Pooling for multi-scale feature aggregation
- **Semantic Segmentation Head**: Predicts pixel-wise semantic classes (19 Cityscapes categories)
- **Instance Embedding Head**: Generates center heatmaps and offset vectors for instance segmentation
- **Panoptic Fusion**: Combines semantic and instance predictions into unified panoptic segmentation

### Outputs[](#outputs)

The model produces:

- **Semantic Logits**: Per-pixel classification scores for semantic classes
- **Center Heatmaps**: Instance center point predictions
- **Offset Vectors**: Pixel-to-instance-center displacement vectors
- **Panoptic Visualization**: Combined semantic and instance segmentation results

## Tracy profiling[](#tracy-profiling)

Tracy is a performance profiling tool that provides visual analysis of execution time and memory usage for each operation during model execution. It is useful for identifying performance bottlenecks and optimizing the PDl model.

Usage:

```
python -m tracy -r -p -v -m pytest models/bos_model/panoptic_deeplab/tests/pcc/test_tt_model.py
```
## L1 Visualizer[](#l1-visualizer)

L1 visualizer is a profiling tool that generates detailed reports on memory usage, buffer allocations, and tensor operations during model execution. It helps in understanding memory consumption patterns and optimizing memory usage for better performance.

- First, export environment variables using the script file. The `experiment_name`can be anything, for example`pdl`.

```
source models/bos_model/export_l1_vis.sh $EXPERIMENT_NAME
```
- Second, run the model. If the model finishes running successfully, the result report will be generated in `generated/ttnn/reports/EXPERIMENT_NAME_MMDD_hhmm/`.

```
pytest models/bos_model/panoptic_deeplab/tests/pcc/test_tt_model.py
```
- Third, run `ttnn-visualizer`. The`report_path`is the path mentioned in the previous step. Open your web browser and navigate to the local address.`http://localhost:8000/`.

```
ttnn-visualizer --profiler-path $REPORT_PATH
```
- If the experiment has finished, please run the following command to clear the environment variables

```
source models/bos_model/unset_l1_vis.sh
```

## 이미지

![BOS_Logo_4E_RGB.avif](../../_assets/BOS_Logo_4E_RGB.avif)  <!-- https://developers.bos-semi.com/img/BOS_Logo_4E_RGB.avif -->
