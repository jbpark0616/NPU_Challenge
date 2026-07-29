---
title: Untitled
source: https://developers.bos-semi.com/docs/Validated_Models/LLM/LLaMA-3.2/
---

# Llama-3.2

Comprehensive guide for running Meta's LLaMA 3.2 language models (1B, 3B, and 8B variants) on BOS Eagle-N hardware. Covers Hugging Face setup, authentication, model execution, batch processing, and performance analysis using profiling tools.

## Introduction[](#introduction)

This codebase includes the Llama-3.2 family of models and currently supports the following variants:

- **Llama-3.2-1B-Instruct:**- [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- **Llama-3.2-3B-Instruct:**- [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **Llama-3.1-8B-Instruct:**- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## Prerequisite: Hugging Face Setup[](#prerequisite-hugging-face-setup)

Before running any Llama model, you must set up your Hugging Face account and access tokens properly.

### 1. Sign Up for Hugging Face[](#1-sign-up-for-hugging-face)

Create an account at [https://huggingface.co](https://huggingface.co) if you don’t already have one.

### 2. Request Access to Llama Models[](#2-request-access-to-llama-models)

Access to Meta’s Llama models is **restricted** — you must manually request permission for each model family.

On each model page, click **“Request Access”** and wait for Meta’s approval.
You’ll receive an email once access is granted.

### 3. Generate an Access Token[](#3-generate-an-access-token)

Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

- Click **“New token”**
- Give it a name (e.g. `llama32-access`)
- Set permission to **Read**
- Copy the token string — you’ll need it in the next step.

### 4. Set Your Token as an Environment Variable[](#4-set-your-token-as-an-environment-variable)

Store your token so it can be used automatically by scripts:

```
export HUGGING_FACE_HUB_TOKEN=your_token_here
```
### 5. Log In via CLI (Alternative)[](#5-log-in-via-cli-alternative)

If you prefer to log in interactively instead of setting an environment variable, use the Hugging Face CLI:

```
huggingface-cli login
```
When prompted, paste your token and press Enter. Once authenticated, your credentials will be stored locally and used automatically for future Hugging Face operations.

### 6. Verify Login[](#6-verify-login)

To confirm that your login or token setup is working correctly, run:

```
huggingface-cli whoami
```
If configured properly, your Hugging Face username will be displayed. If it shows an authentication error, please repeat Step 5 or check your token’s permission settings.

## Set environment variables[](#set-environment-variables)

```
# at $TT_METAL_HOME
source env_set.sh
```
## How to Run[](#how-to-run)

For a single user example:

```
HF_MODEL=model_name pytest models/bos_model/llama32/demo/demo.py -k 'performance and batch-one'
```
**Notes:**

- `model_name`is the HuggingFace model repo string, e.g.- `meta-llama/Llama-3.2-1B-Instruct`or- `meta-llama/Llama-3.2-3B-Instruct`.
- `-k`is the pytest filter; to run a specific test, use- `-k test_name`; additional test names are listed in- `models/bos_model/llama32/demo/demo.py`

For a batch user example:

```
HF_MODEL=model_name pytest models/bos_model/llama32/demo/demo.py -k 'performance and batch-16'
```
For Live chatting demo example, using Llama-3.2-1B:

```
HF_MODEL=meta-llama/Llama-3.2-1B-Instruct python models/bos_model/llama32/demo_llama32.py
```
**Notes:**

- Use `--no-memory`to disable Llama remembering context.
- Use `-g <integer>`to set the maximum generated tokens per answer.
- use `--test`to make Llama use 4 preset questions instead of entering questions yourself.
- Loading the 8B model may take some time. It is recommended to wait 1–2 minutes for the model to fully initialize before starting the chat.

## Details[](#details)

- On the first execution of each model, TTNN will create weight cache files for that model, to speed up future runs. These cache files only need to be created once for each model and each weight (i.e. new finetuned weights will need to be cached) and will be stored accordingly to the machine you are running the models.

## Tracy profiling[](#tracy-profiling)

Tracy is a performance profiling tool that provides visual analysis of execution time and memory usage for each operation during model execution.

Just make sure you have built Metal using the `-p` tag, instead of `-b Release` tag, to enable profiling

Usage:

```
HF_MODEL=model_name python -m tracy -r -p -v -m pytest models/bos_model/llama32/test_demo_llama32.py
```
## Run for ttnn-visualizer Profiler[](#run-for-ttnn-visualizer-profiler)

- First, export ENV using script file
- `$EXPERIMENT_NAME`: input anything (for example,- `llama`)
 

```
source models/bos_model/export_l1_vis.sh $EXPERIMENT_NAME
```
- Second, run model
- If the model has finished running successfully, the result report will be generated in the following path (`generated/ttnn/reports/EXPERIMENT_NAME_MMDD_hhmm/`)
 
- If the model has finished running successfully, the result report will be generated in the following path (

```
HF_MODEL=model_name pytest models/bos_model/llama32/test_demo_llama32.py
```
- Third, run ttnn-visualizer
- `$REPORT_PATH`: It is the path mentioned in the previous step
- visit `http://localhost:8000/`using your web browser
 

```
ttnn-visualizer --profiler-path $REPORT_PATH
```
- If the experiment has finished, please run the following command to clear the environment variables

```
source models/bos_model/unset_l1_vis.sh
```

## 이미지

![BOS_Logo_4E_RGB.avif](../../_assets/BOS_Logo_4E_RGB.avif)  <!-- https://developers.bos-semi.com/img/BOS_Logo_4E_RGB.avif -->
