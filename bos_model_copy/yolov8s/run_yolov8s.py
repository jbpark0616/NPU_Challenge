import copy
import importlib
import math
import os

import cv2
import supervision as sv
import torch
import torch.nn as nn
from ultralytics.engine.results import Results
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.block import DFL
from ultralytics.utils import ops
from ultralytics.utils.tal import dist2bbox, make_anchors

import ttnn

torch.set_printoptions(precision=2, threshold=100)

debug_printer = False

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import argparse
from time import sleep, time

import pytest
from reference.reference_model import YOLOv8_AlignedWithPT as Torch_YOLO
from utilities.utility_functions import (
    _nearest_32,
    comp_pcc,
    load_resize_and_pad_channels,
    setup_l1_sharded_input,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from yolov8s import Detect, YoloV8, class_names


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num_images", type=int, default=3, help="number of images to process")
    parser.add_argument("-i", "--image_size", type=int, default=256, help="image size processed by bos model")
    parser.add_argument("--trace", default=False, action="store_true", help="enable trace mode")
    parser.add_argument("-p", "--persistent_cache", default=False, action="store_true", help="enable trace mode")

    args, _ = parser.parse_known_args(argv)
    return args


def main(device, args):
    device.enable_program_cache()
    if args.persistent_cache:
        ttnn.device.EnablePersistentKernelCache()

    images = [
        "16004479832_a748d55f21_k.jpg",
        "17790319373_bd19b24cfc_k.jpg",
        "18124840932_e42b3e377c_k.jpg",
        "19064748793_bb942deea1_k.jpg",
        "24274813513_0cfd2ce6d0_k.jpg",
        "33823288584_1d21cf0a26_k.jpg",
        "33887522274_eebd074106_k.jpg",
    ]
    images = images[: args.num_images]  # Limit number of images to process

    input_dir = os.path.join(current_dir, "reference/images")

    batch_size = 1
    image_height = args.image_size
    image_width = args.image_size
    input_channels = 3
    output_classes = 80

    layer_config_file = f"models.bos_model.yolov8s.configs.yolov8s_{image_height}x{image_width}"
    if not os.path.exists(os.path.join(current_dir, f"configs/{layer_config_file.split('.')[-1]}.py")):
        layer_config_file = f"models.bos_model.yolov8s.configs.yolov8s_256x256"
        print(f"Using {layer_config_file} as config file")
    layer_configs = importlib.import_module(layer_config_file).layer_configs

    bos_model = YoloV8(
        device=device,
        image_shape=[image_height, image_width],
        in_channels=input_channels,
        num_classes=output_classes,
        layer_configs=layer_configs,
    )
    bos_model.eval()

    torch_model = Torch_YOLO()
    torch_model.load_ckpt(os.path.join(current_dir, "yolov8s.pt"))
    torch_model.eval()

    time_values = []

    # Supervision Annotator 설정
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.6, text_padding=1)

    # Set torch Detection layer
    torch_detect_layer = Detect(nc=80, ch=[128, 256, 512])
    torch_detect_weights = torch.load(os.path.join(current_dir, "detect_weights.pth"))
    torch_detect_layer.load_state_dict(torch_detect_weights)
    torch_detect_layer.eval()

    if args.trace:  # Warm up the model
        print("Warming up and Capturing the model for trace mode")
        torch_image, original_size, input_size = load_resize_and_pad_channels(
            os.path.join(input_dir, images[0]), image_size=(image_height, image_width), golden=True
        )
        test_image_host, input_mem_config = setup_l1_sharded_input(device, torch_image)

        # Dummy event recording
        op_event = ttnn.record_event(device, 0)

        # Sending input to Device
        ttnn.wait_for_event(1, op_event)
        bos_model.input_tensor = test_image_host.to(device, input_mem_config)
        write_event = ttnn.record_event(device, 1)

        # Warm up
        ttnn.wait_for_event(0, write_event)
        op_event = ttnn.record_event(device, 0)
        outputs = bos_model()

        # Capture
        start_time = time()
        ttnn.wait_for_event(1, op_event)
        write_event = ttnn.record_event(device, 1)
        op_event = ttnn.record_event(device, 0)
        input_trace_addr = bos_model.input_tensor.buffer_address()
        spec = bos_model.input_tensor.spec
        # output_tensor.deallocate(force=True)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        outputs = bos_model()
        # input_l1_tensor = ttnn.allocate_tensor_on_device(spec, device)
        # assert input_trace_addr == input_l1_tensor.buffer_address()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        elapsed_time = time() - start_time

    for i in range(args.num_images):
        image = images[i % len(images)]
        torch_image, original_size, input_size = load_resize_and_pad_channels(
            os.path.join(input_dir, image), image_size=(image_height, image_width), golden=True
        )
        torch_outputs = torch_model(torch_image)  # returns ([detect_22], [p3_td, p4_out, p5_out])

        start_time = time()
        if args.trace:
            ttnn.wait_for_event(1, op_event)
            test_image_host, input_mem_config = setup_l1_sharded_input(device, torch_image)
            ttnn.copy_host_to_device_tensor(test_image_host, bos_model.input_tensor, 0)
            write_event = ttnn.record_event(device, 1)
            ttnn.wait_for_event(0, write_event)
        else:
            # test_image, original_size, input_size = load_resize_and_pad_channels(
            #     os.path.join(input_dir, image), image_size=(image_height, image_width)
            # )
            test_image, input_mem_config = setup_l1_sharded_input(device, torch_image)
            test_image = test_image.to(device, input_mem_config)

        if args.trace:
            op_event = ttnn.record_event(device, 0)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        else:
            outputs = bos_model(test_image)
        output0 = ttnn.from_device(bos_model.output0, blocking=True)
        output1 = ttnn.from_device(bos_model.output1, blocking=True)
        output2 = ttnn.from_device(bos_model.output2, blocking=True)
        outputs = [output0, output1, output2]
        ttnn.synchronize_device(device)

        elapsed_time = time() - start_time
        if args.trace:
            print(f"[{i}](Trace) Time taken ={elapsed_time:.4f} s")
            print(f"[{i}](Trace) FPS = {(1 / (elapsed_time)):.2f} Hz")
        else:
            print(f"[{i}] Time taken = {elapsed_time:.4f} s")
            print(f"[{i}] FPS = {(1 / (elapsed_time)):.2f} Hz")

        detect_heads = []
        for output, golden_output in zip(outputs, torch_outputs[0]):
            torch_output = tt_to_torch_tensor(output)
            # ttnn.deallocate(output)
            # pad_value = torch_output.shape[2] - (golden_output.shape[2] * golden_output.shape[3])
            # if pad_value > 0:
            #     torch_output = torch_output[:, :, :-pad_value, :]
            torch_output = torch_output.permute(0, 3, 1, 2)
            torch_output = torch_output.reshape(golden_output.shape).to(torch.float32)
            # _, pcc = comp_pcc(golden_output, torch_output)
            # print(f"\tPCC = {pcc:.3f}")
            # assert pcc > 0.995
            detect_heads.append(torch_output)

        # final_output = torch_detect_layer(detect_heads)
        final_output = torch_detect_layer.post_detect_inference(detect_heads)
        print("Final FPS = ", 1 / (time() - start_time))
        nms_result = ops.non_max_suppression(final_output[0], conf_thres=0.25, iou_thres=0.45)

        if args.trace:
            elapsed_time_e2e = time() - start_time
            print(f"[{i}](Trace-e2e) Time taken = {elapsed_time_e2e:.4f} s")
            print(f"[{i}](Trace-e2e) FPS = {(1 / (elapsed_time_e2e)):.2f} Hz")

        for _, det in enumerate(nms_result):
            if det is not None and len(det):
                boxes = det[:, :4].detach().cpu().numpy()
                confs = det[:, 4].detach().cpu().numpy()
                class_ids = det[:, 5].detach().cpu().numpy().astype(int)

                orig_w, orig_h = original_size
                input_w, input_h = input_size
                scale_x, scale_y = orig_w / input_w, orig_h / input_h
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

                detections = sv.Detections(xyxy=boxes, confidence=confs, class_id=class_ids)
                labels = [f"{class_names.get(cls, str(cls))} {conf:.2f}" for cls, conf in zip(class_ids, confs)]

                original_image = cv2.imread(os.path.join(input_dir, image))
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                annotated_image = box_annotator.annotate(scene=original_image, detections=detections)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

                annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

                # annotated_bgr = det.plot()
                filename = os.path.basename(image)
                os.makedirs(os.path.join(current_dir, "results"), exist_ok=True)
                out_path = os.path.join(current_dir, f"results/boxed_{filename}")
                cv2.imwrite(out_path, annotated_bgr)
                print(f"✅ Saved result: {out_path}\n")
            else:
                print(f"No detections found for image {image} <- len(det)={len(det)}\n")


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 10240, "trace_region_size": 3686400, "num_command_queues": 2}], indirect=True
)
def test_trace_main(device):
    args = parse_args()
    args.trace = True
    main(device, args)


@pytest.mark.parametrize("l1_small_size", [10240])
def test_main(l1_small_size):
    device = ttnn.open_device(device_id=0, l1_small_size=l1_small_size)
    args = parse_args([])  # 또는 args = parse_args()
    main(device, args)


if __name__ == "__main__":
    args = parse_args()

    if args.trace:
        # device = ttnn.open_device(device_id=0, l1_small_size=20480, trace_region_size=10419200, num_command_queues=2)
        device_dict = {"device_id": 0, "l1_small_size": 20480, "trace_region_size": 10419200, "num_command_queues": 2}
        ttnn_device = ttnn._ttnn.device
        device = ttnn_device.CreateDevice(**device_dict)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=20480)
    main(device, args)
    ttnn.close_device(device)
