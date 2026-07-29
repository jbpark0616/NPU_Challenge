---
title: Untitled
source: https://developers.bos-semi.com/docs/Validated_Models/LLM/Qwen2.5-VL/
---

# Qwen2.5-VL

Complete guide for deploying the Qwen2.5-VL multimodal model (3B and 7B variants) on BOS Eagle-N hardware. This guide covers model setup, Hugging Face authentication, inference execution, batch processing, and performance profiling with Tracy.

## Introduction[](#introduction)

This codebase includes the Qwen2.5-VL family of models and currently supports the model variants:

- Qwen2.5-VL-3B-Instruct: [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Qwen2.5-VL-7B-Instruct: [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Qwen2.5-VL-3B-Instruct-AWQ: [Qwen/Qwen2.5-VL-3B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ)
- Qwen2.5-VL-7B-Instruct-AWQ: [Qwen/Qwen2.5-VL-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ)

## Set environment variables[](#set-environment-variables)

```
# at $TT_METAL_HOME
source env_set.sh
```
## How to Run[](#how-to-run)

### For a single user example:[](#for-a-single-user-example)

```
HF_MODEL=<model_name> pytest models/bos_model/qwen25_vl/demo/vision_demo.py -k 'accuracy and batch1-trace'
```
**Notes:**

- `<model_name>`is the HuggingFace model repo string, e.g.- `Qwen/Qwen2.5-VL-3B-Instruct`,- `Qwen/Qwen2.5-VL-7B-Instruct`,- `Qwen2.5-VL-3B-Instruct-AWQ`, or- `Qwen2.5-VL-7B-Instruct-AWQ`.
- `-k`is the pytest filter; to run a specific test, use- `-k <test_name>`; additional test names are listed in- `models/bos_model/qwen25_vl/demo/vision_demo.py`.
- `models/bos_model/qwen25_vl/demo/outputs`is the path to the directory containing dumped vision outputs.
- `--res`is an optional flag to specify the input resolution for vision tests. It currently supports- `128x128`and- `224x224`, and defaults to- `224x224`.

### For a batch user example:[](#for-a-batch-user-example)

```
HF_MODEL=<model_name> pytest models/bos_model/qwen25_vl/demo/vision_demo.py -k 'accuracy and batch2-trace'
```
- Note: The current implementation supports a batch size of 2.

### To capture Tracy report:[](#to-capture-tracy-report)

```
HF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct python -m tracy -m -r -p -v "pytest models/bos_model/qwen25_vl/demo/vision_demo.py -k 'accuracy and profiler'"
```
**Notes:**

- The model name `Qwen/Qwen2.5-VL-7B-Instruct`can be changed to`Qwen/Qwen2.5-VL-3B-Instruct`if you want to record Tracy for the 3B model.
- `-k`is the pytest filter. To run a specific test,- `profiler`is a special test case for Tracy recording or- `ttnn-visualizer`. Use this test case for profiling.
- Profiling parameters: `res = [224, 224]`,`max_batch_size = 1`,`warmup_iters = 0`,`include_text_only_prompts = False`.
- The `accuracy`mode (BF16) can be changed to`performance`(BFP8-mixed).
- `generated/profiler/reports`is the path to the directory containing the Tracy report. Refer to- [tt-perf-report](https://github.com/tenstorrent/tt-perf-report)to read the report.

### For Live chatting demo example:[](#for-live-chatting-demo-example)

- `HF_MODEL=<model_name> python models/bos_model/qwen25_vl/demo_qwen25_vl.py -i <image_path>`

**Notes:**

- Use `-i`to pass the input image path to the model, for example`models/bos_model/qwen25_vl/demo/images/dog.jpg`.
- Use `-c`to enable Qwen to remember context.
- Since image tokens are large, the context will grow with each interaction. Over time, this can exceed memory limits, so for longer chats it is recommended to run Qwen without context.

## Details[](#details)

- On the first execution of each model, TTNN will create weight cache files for that model to speed up future runs.
- These cache files only need to be created once for each model and each weight. New fine-tuned weights will need to be cached separately and will be stored according to the machine you are running the models on.

## Run for ttnn-visualizer Profiler[](#run-for-ttnn-visualizer-profiler)

- First, export environment variables using the script file.
- `$EXPERIMENT_NAME`: input any string, for example- `qwen`
 

```
source models/bos_model/export_l1_vis.sh $EXPERIMENT_NAME
```
- Second, run the model.
- If the model finishes running successfully, the result report will be generated in `generated/ttnn/reports/$EXPERIMENT_NAME_MMDD_hhmm/`.
 
- If the model finishes running successfully, the result report will be generated in 

```
HF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct pytest models/bos_model/qwen25_vl/demo/vision_demo.py -k 'accuracy and profiler'
```
- Third, run `ttnn-visualizer`.- `$REPORT_PATH`: the path mentioned in the previous step
- Visit `http://localhost:8000/`using your web browser
 

```
ttnn-visualizer --profiler-path $REPORT_PATH
```
- If the experiment has finished, run the following command to clear the environment variables.

```
source models/bos_model/unset_l1_vis.sh
```

## 이미지

![BOS_Logo_4E_RGB.avif](../../_assets/BOS_Logo_4E_RGB.avif)  <!-- https://developers.bos-semi.com/img/BOS_Logo_4E_RGB.avif -->
