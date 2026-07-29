---
title: Untitled
source: https://developers.bos-semi.com/docs/AI_Models_SDK/Custom_ops_development/
---

# Custom Ops Development

Learn how to extend TT-NN by implementing custom neural network operators for your specific workloads. This guide covers creating device operations and composite operations in C++, registration, and exposing Python bindings for integration with the TT-NN framework.

# Custom Operators

[https://docs.tenstorrent.com/tt-metal/v0.55.0/ttnn/ttnn/adding_new_ttnn_operation.html](https://docs.tenstorrent.com/tt-metal/v0.55.0/ttnn/ttnn/adding_new_ttnn_operation.html)

### Create Custom TT-NN Operations[](#create-custom-tt-nn-operations)

The **TT-NN** library supports **extending its built-in operator set** by letting developers define **custom neural network operations** that can be invoked just like the standard ones.

You can implement a new operator in **C++** either as:

- A **device operation**(providing the logic and program to run on the accelerator)
- A **composite operation**that builds on other existing ops

Once implemented, you register the operation so that it’s callable from the **TT-NN API**. You can also expose **Python bindings** for your custom operator so it integrates seamlessly with Python applications.

This extensibility makes it easy to tailor TT-NN to support new primitives or experimental kernels specific to your workloads.

## 이미지

![BOS_Logo_4E_RGB.avif](../_assets/BOS_Logo_4E_RGB.avif)  <!-- https://developers.bos-semi.com/img/BOS_Logo_4E_RGB.avif -->
