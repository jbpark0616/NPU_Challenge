---
title: Untitled
source: https://developers.bos-semi.com/docs/AI_Models_SDK/Tools/tracy/
---

# Performance Profiler

Performance profiler is built on top of Tracy which is an open-source C++ profiling tool with sampling and code instrumentation profiling capabilities.

- **Tracy**is an open-source- **C++ profiling tool**with both- **sampling**and- **code instrumentation**profiling capabilities.
- Tracy can profile all host-side code in tt_metal, including Python.
- The **profiling results**for all cores’**RISCs**can be viewed**graphically**.

Figure 1. Screen capture of Tracy Profiling Tool

## Tensix Core Architecture[](#tensix-core-architecture)

### Data Movement[](#data-movement)

- **BRISC**: Reads- **weights**from- **DRAM → Circular Buffer (CB)**
- **NRISC**: Reads- **activations**from- **L1 → CB**

### Instruction Dispatch to Compute Engine[](#instruction-dispatch-to-compute-engine)

- **TRISC0**: For- **unpacker**to read- **weight and activation data**(- **CB → Src register**)
- **TRISC1**: For- **compute**to perform- **matrix multiplication (matmul)**
- **TRISC2**: For- **packer**to write- **results**to- **CB**(- **Dst register → CB**)

Figure 2. Tensix Core Architecture

## Profiling with Tracy – (1) Source Code Edit[](#profiling-with-tracy--1-source-code-edit)

To profile the **matmul** operation, edit the following source file:

`ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`

- **Include the header and enable logging**- `#define DEBUG_TRACY_LOG 1`
 #if DEBUG_TRACY_LOG
 #include "tools/profiler/kernel_profiler.hpp"
 #endif
- **making a zone**- `void MAIN() {`
 ...
 {
 #if DEBUG_TRACY_LOG
 DeviceZoneScopedN("matmul_block");
 #endif
 for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
 matmul_block(
 in0_cb_id, in1_cb_id, in0_index, in1_index, dst_index,
 in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w
 );
 in0_index++;
 in1_index += in1_block_w;
 }
 }
 }

## Profiling with tracy – (2) build and run[](#profiling-with-tracy--2-build-and-run)

### At the server[](#at-the-server)

- **1. build with profiler option**- `$ cd tt-metal`
 $ build_metal.sh --enable-profiler
- **2. build with profiler option**- `$ ENABLE_TRACY=1 TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -p -v -m pytest models/demos/wormhole/vit/demo/demo_vit_ttnn_inference_perf_e2e_2cq_trace.py`
- **3. then tracy file is generated at ‘tt-metal/generated/profiler/.logs/tracy_profile_log_host.tracy'**
- **4. download the tracy file to client**

### At the client[](#at-the-client)

- **1. run tracy**
- **2. ‘Open saved trace’ and open downloaded tracy file**

Figure 3. Tracy Control Box

## Profiling with tracy – (3) matmul analysis[](#profiling-with-tracy--3-matmul-analysis)

- **TRISC0 waits until the activations and weights are loaded into the circular buffer**

Figure 4. Analysis of MatMul Operation

## Profiling with tracy – (4) matmul analysis[](#profiling-with-tracy--4-matmul-analysis)

- **TRISC1 waits until the unpacker fetches activations and weights into the source register**
- **TRISC2 waits until the math result is saved to the destination register**

Figure 5. Analysis of MatMul Operation

## Profiling with tracy – (5) Softmax analysis[](#profiling-with-tracy--5-softmax-analysis)

- **attention and softmax are fused**
- **1. scale attention score → [multiply]**- **load the scale value from DRAM, and multiply it with the attention score matrix**
 
- **2. masking : mask + scaled attention score → [addition]**- **TRISC0 waits until scale multiply is done and mask is loaded from NCRISC**
 

Figure 6. Analysis of Softmax Operation

## Profiling with tracy – (6) Softmax analysis[](#profiling-with-tracy--6-softmax-analysis)

- **2. masking : mask + scaled attention score → [addition]**- **TRISC0 send instruction for unpacker**
- **TRISC1 send instruction for math to add, and result will be saved in dst register**
 
- **3. exp(x) → [exponential]**- **TRISC1 send instruction for exponential**
- **TRISC2 send instruction for packer, to save exponential result to CB**
- **TRISC0 wait until TRISC2 write the result to CB, then proceed to ‘sum’**
 

Figure 6. Analysis of Softmax Operation

## Profiling with tracy – (7) Softmax analysis[](#profiling-with-tracy--7-softmax-analysis)

- **4. sum(exp(x)) → [sum]**- unpack and matmul to sum the exponential results
 
- **5. 1/sum(exp(x)) → [recip]**- after performing the matmul for sum, the SFPUs calculate the reciprocal of ‘sum(exp(x))
 
- **6. exp(x) / sum(exp(x)) → [mul]**- **TRISC0**waits until packer saves the ‘1/sum(exp(x))’, then multiply with exp(x)
 

Figure 7. Analysis of Softmax Operation

## Additional Resources[](#additional-resources)

[Tenstorrent Tracy Profiler How to Use](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html)

[How to use DPRINT to see how the device works](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tools/kernel_print.rst)

## 이미지

![matmul_analysis_3.jpg](../../_assets/matmul_analysis_3.jpg)  <!-- https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/matmul_analysis_3.jpg -->
![screen_capture_tracy.jpg](../../_assets/screen_capture_tracy.jpg)  <!-- https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/screen_capture_tracy.jpg -->
![softmax_analysis_7.jpg](../../_assets/softmax_analysis_7.jpg)  <!-- https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/softmax_analysis_7.jpg -->
![BOS_Logo_4E_RGB.avif](../../_assets/BOS_Logo_4E_RGB.avif)  <!-- https://developers.bos-semi.com/img/BOS_Logo_4E_RGB.avif -->
![softmax_analysis_5.jpg](../../_assets/softmax_analysis_5.jpg)  <!-- https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/softmax_analysis_5.jpg -->
![tracy_controlbox.jpg](../../_assets/tracy_controlbox.jpg)  <!-- https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/tracy_controlbox.jpg -->
![matmul_analysis_4.jpg](../../_assets/matmul_analysis_4.jpg)  <!-- https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/matmul_analysis_4.jpg -->
![softmax_analysis_6.jpg](../../_assets/softmax_analysis_6.jpg)  <!-- https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/softmax_analysis_6.jpg -->
![tensix_core_arch.jpg](../../_assets/tensix_core_arch.jpg)  <!-- https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/tensix_core_arch.jpg -->
