# for display
import argparse
import multiprocessing

# import matplotlib
# matplotlib.use("QtAgg")
import cv2
import time

import os
import torch
# import matplotlib.pyplot as plt
from loguru import logger
logger.remove()  # Removes the default stdout sink

import ttnn
from models.common.utility_functions import profiler
from models.perf.perf_utils import prep_perf_report
from tests.ttnn.unit_tests.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
from tests.ttnn.utils_for_testing import check_with_pcc

from models.bos_model.fastoft.reference.encoder import ObjectEncoder

# Import OFT components
from models.bos_model.fastoft.reference.oftnet import FrontendMode, OftMode, OftNet
from models.bos_model.fastoft.reference.utils import load_calib, load_image, make_grid
from models.bos_model.fastoft.tests.common import GRID_HEIGHT, GRID_RES, GRID_SIZE, H_PADDED, NMS_THRESH, W_PADDED, Y_OFFSET, load_checkpoint
from models.bos_model.fastoft.tt.model_configs import ModelOptimizations
from models.bos_model.fastoft.tt.model_preprocessing import (
    create_decoder_model_parameters,
    create_OFT_model_parameters,
    fuse_conv_bn_parameters,
    fuse_imagenet_normalization,
)
from models.bos_model.fastoft.tt.oft_pipeline import PipelineConfig, create_pipeline_from_config
from models.bos_model.fastoft.tt.tt_encoder import TTObjectEncoder
from models.bos_model.fastoft.tt.tt_oftnet import TTOftNet
from models.bos_model.fastoft.tt.tt_resnet import TTBasicBlock
from models.bos_model.fastoft.reference.bbox import visualize_objects_cv

def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n", "--num_iter", type=int, default=10, help="number of iteration to process. -1 is infinite loop"
    )
    parser.add_argument("-i", "--input", type=str, nargs="+", default=os.path.join(os.path.dirname(__file__), "resources"), help="input image path")
    parser.add_argument("-c", "--calib", type=str, nargs="+", default=os.path.join(os.path.dirname(__file__), "resources"), help="calibration file path")

    args, _ = parser.parse_known_args(argv)
    return args

def cv2_worker(lock, shared_dict):
    while True:
        with lock:
            output_image = shared_dict["output_image"]
        if output_image is not None:
            cv2.imshow("OFT", output_image)
            if not shared_dict["running"]:  # run all images
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(10)
            if key == 27 or key == ord("q"):  # ESC or 'q'
                shared_dict["running"] = False
                break
    cv2.destroyAllWindows()


class OFTPerformanceRunnerInfra:
    """
    OFT test infrastructure - encapsulates model setup and execution.

    Following ResNet test_infra pattern.
    """

    def __init__(self, device, input_tensor, calib, grid, ref_model, oft_mode, model_location_generator):
        self.device = device
        self.oft_mode = oft_mode
        self.batch_size = input_tensor.shape[0]

        # Preprocessing (demo.py pattern)
        logger.info("[Infra] Applying preprocessing...")
        fuse_imagenet_normalization(ref_model, ref_model.mean, ref_model.std)

        state_dict = create_OFT_model_parameters(ref_model, (input_tensor, calib, grid), device=device)
        state_dict = {"oftnet": state_dict}
        state_dict = fuse_conv_bn_parameters(state_dict)["oftnet"]

        model_opt = ModelOptimizations()
        model_opt.apply(state_dict)

        # Create TT model
        logger.info("[Infra] Creating TT model...")
        self.tt_model = TTOftNet(
            device,
            state_dict,
            state_dict.layer_args,
            TTBasicBlock,
            [2, 2, 2, 2],
            ref_model.mean,
            ref_model.std,
            input_shape_hw=(H_PADDED, W_PADDED),
            calib=calib,
            grid=grid,
            topdown_layers=8,  # Fixed for OFT8/REDUCED
            grid_res=GRID_RES,
            grid_height=GRID_HEIGHT,
            oft_mode=oft_mode,
            preprocess_frontend_conv1=True,
        )

        # Pre-convert persistent tensors
        self.tt_calib = ttnn.from_torch(calib, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.tt_grid = ttnn.from_torch(grid, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        grid_ = grid.clone().squeeze(0)
        self.tt_grid_ = ttnn.from_torch(grid_, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        # Create reference encoder for decoder setup
        ref_encoder = ObjectEncoder(nms_thresh=NMS_THRESH, dtype=torch.float32)

        # Create dummy decoder inputs for parameter initialization
        # Grid determines the spatial dimensions (160x160 for GRID_SIZE=80, GRID_RES=0.5)
        # Shapes after squeezing batch dim match grid spatial dims
        grid_h, grid_w = grid_.shape[0], grid_.shape[1]
        dummy_scores = torch.zeros(1, grid_h - 1, grid_w - 1)
        dummy_pos_offsets = torch.zeros(1, 3, grid_h - 1, grid_w - 1)
        dummy_dim_offsets = torch.zeros(1, 3, grid_h - 1, grid_w - 1)
        dummy_ang_offsets = torch.zeros(1, 2, grid_h - 1, grid_w - 1)
        decoder_params = create_decoder_model_parameters(
            ref_encoder, [dummy_scores, dummy_pos_offsets, dummy_dim_offsets, dummy_ang_offsets, grid_], device
        )

        # Create TT encoder (decoder)
        logger.info("[Infra] Creating TT encoder (decoder)...")
        self.tt_encoder = TTObjectEncoder(device, decoder_params, grid_, nms_thresh=NMS_THRESH)

        # Store input/output for model_wrapper
        self.input_tensor = None
        self.output_tensor = None

        # Store reference output for validation
        self.ref_output = None

    def setup_dram_sharded_input(self, device):
        """
        Setup DRAM and L1 sharded input configuration.

        Following ResNet pattern (resnet50_test_infra.py lines 263-298).
        Creates proper HEIGHT_SHARDED L1 config for traced execution.
        """
        # Create preprocessed tensor on host first
        # OFT uses [B, H, W, C] layout after permute
        tt_input = self.input_tensor_host.permute((0, 2, 3, 1))  # [1, 384, 1280, 3]
        # DRAM interleaved config
        dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Create host tensor with proper layout
        tt_input_host = ttnn.from_torch(
            tt_input,
            dtype=ttnn.bfloat16,
            # layout=ttnn.ROW_MAJOR_LAYOUT,
            layout=ttnn.TILE_LAYOUT,
        )

        return tt_input_host, dram_memory_config, dram_memory_config

    def preprocess(self, input_tensor):
        self.input_tensor = input_tensor
        self.preprocessed_tensor = self.tt_model.preprocess(self.input_tensor)
        return self.preprocessed_tensor

    def run(self, input_tensor):
        """Run OFT model forward pass + decoder"""
        self.input_tensor = input_tensor
        (intermediates, layer_names), tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = self.tt_model.forward(
            self.device, self.input_tensor, self.tt_calib, self.tt_grid
        )

        # Run decoder (following demo.py pattern)
        tt_scores = ttnn.to_layout(ttnn.squeeze(tt_scores, 0), layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_pos_offsets = ttnn.to_layout(ttnn.squeeze(tt_pos_offsets, 0), layout=ttnn.TILE_LAYOUT)
        tt_dim_offsets = ttnn.to_layout(ttnn.squeeze(tt_dim_offsets, 0), layout=ttnn.TILE_LAYOUT)
        tt_ang_offsets = ttnn.to_layout(ttnn.squeeze(tt_ang_offsets, 0), layout=ttnn.TILE_LAYOUT)

        # decode returns: (tt_outs_list, intermediates_list, names_tuple, intermediate_names_tuple)
        # We only need the first element for the pipeline
        tt_outs, _, _, _ = self.tt_encoder.decode(
            self.device, tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets
        )
        # tt_outs is [tt_indices, tt_scores, tt_positions, tt_dimensions, tt_angles]

        # Convert to tuple for pipeline compatibility
        self.output_tensor = tuple(tt_outs)
        return self.output_tensor

    def postprocess(self, output):
        """Postprocess decoder outputs (torch code, not traced)"""
        # output is [tt_indices, tt_scores, tt_positions, tt_dimensions, tt_angles]
        # Convert TTNN outputs to torch tensors
        tt_outs_torch = self.tt_encoder.decoder_postprocess(*output)
        # Return scores for validation
        return tt_outs_torch

    def validate(self, tt_outs_torch, expected_detections=None):
        """No need here for demo image run"""
        pass
        # """Validate outputs against number of detected objects"""
        # tt_objects_torch = self.tt_encoder.create_objects(*tt_outs_torch)
        # logger.info("=== TTNN Objects ===")
        # detected = len(tt_objects_torch)
        # if expected_detections is not None:
        #     assert detected == expected_detections, f"Expected {expected_detections} detections, got {len(tt_objects_torch)}"
        # for i, obj in enumerate(tt_objects_torch):
        #     logger.info(f"TT Object {i}: {obj}")

        # return True, f"Validation passed. {detected=} {expected_detections=} objects detected."


def main(device, model_location_generator, input_image_paths, calib_paths, args):
    model_dtype=torch.float32
    oft_mode = OftMode.OFT8
    frontend_mode = FrontendMode.REDUCED

    profiler.clear()
    device.disable_and_clear_program_cache()

    # Start inference thread
    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    shared_dict = manager.dict(
        {
            "running": True,
            "output_image": None,
        }
    )
    cv2_thread = multiprocessing.Process(target=cv2_worker, args=(lock, shared_dict))

    # ========================================================
    # OFT model configuration based on real model parameters
    print(f"** Generating Torch ref model **")

    # 1 Handle inputs
    print(f"** Loading {len(input_image_paths)} input images for round-robin testing **")
    input_tensors = []
    for img_path in input_image_paths:
        img_tensor = load_image(img_path, pad_hw=(H_PADDED, W_PADDED), dtype=model_dtype)[None].to(model_dtype)
        input_tensors.append(img_tensor)

    # Use first calib/grid for model setup (all should be compatible)
    calib = load_calib(calib_paths[0], dtype=model_dtype)[None].to(model_dtype)
    grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=model_dtype)[None].to(
        model_dtype
    )

    # 2 Create reference OFTnet
    print("** Creating reference model **")
    ref_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        dtype=model_dtype,
        oft_mode=oft_mode,
        frontend_mode=frontend_mode,
    )
    ref_model = load_checkpoint(ref_model, model_location_generator, oft_mode=oft_mode)
    ref_model.eval()

    # Create test infrastructure (use first input for setup)
    print("** Creating ttnn test infrastructure **")
    device.enable_program_cache()
    test_infra = OFTPerformanceRunnerInfra(
        device, input_tensors[0], calib, grid, ref_model, oft_mode, model_location_generator
    )

    ttnn.synchronize_device(device)

    with torch.no_grad():
        # _run_model_pipeline with 2cqs and trace
        _num_command_queues = 1
        _trace = True

        # Use first input for warmup and compilation
        test_infra.input_tensor_host = input_tensors[0]

        # Setup DRAM sharded input
        tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)

        # Create pipeline
        pipeline = create_pipeline_from_config(
            config=PipelineConfig(
                use_trace=_trace, num_command_queues=_num_command_queues, all_transfers_on_separate_command_queue=False
            ),
            model=test_infra,
            device=device,
            dram_input_memory_config=sharded_mem_config_DRAM,
            l1_input_memory_config=None,
        )

        logger.info(f"Running model warmup with input shape {list(tt_inputs_host.shape)}")
        profiler.start("compile")
        pipeline.compile(tt_inputs_host)
        profiler.end("compile")

        num_warmup_iterations = 1
        # Run warmup iterations with first input only
        if num_warmup_iterations > 0:
            logger.info(f"Running {num_warmup_iterations} warmup iterations")
            print(f"** Running {num_warmup_iterations} warmup iterations **")
            warmup_inputs = [tt_inputs_host] * num_warmup_iterations
            pipeline.preallocate_output_tensors_on_host(num_warmup_iterations)
            pipeline.enqueue(warmup_inputs).pop_all()
            print("** Warmup complete **")

        iter = 0
        sum_elapsed_time = 0
        num_iter = args.num_iter if args.num_iter >= 0 else float("inf")
        command_queues_preload_size = 1
        cv2_thread.start()
        while True:
            org_inputs = []
            host_inputs = []
            for _ in range(command_queues_preload_size):
                input_idx = iter % len(input_tensors)
                test_infra.input_tensor_host = input_tensors[input_idx]
                org_inputs.append(input_tensors[input_idx])
                tt_input_host, _, _ = test_infra.setup_dram_sharded_input(device)
                host_inputs.append(tt_input_host)

            # Preallocate outputs
            pipeline.preallocate_output_tensors_on_host(command_queues_preload_size)

            logger.info(f"Starting performance pipeline for {command_queues_preload_size} iterations")
            profiler.start(f"run_model_pipeline_{_num_command_queues}cqs")
            outputs = pipeline.enqueue(host_inputs).pop_all()
            profiler.end(f"run_model_pipeline_{_num_command_queues}cqs")
            iter += 1
            inference_time = profiler.get(f"run_model_pipeline_{_num_command_queues}cqs")
            print(f"[{iter}] images processed avg FPS: {(command_queues_preload_size/inference_time):.2f} Hz")

            # Visualize predictions
            for i, (_input, _output) in enumerate(zip(org_inputs,outputs)):
                torch_outputs = test_infra.postprocess(_output)
                tt_objects_torch = test_infra.tt_encoder.create_objects(*torch_outputs)
                _input = _input.to(torch.float32).squeeze(0)
                calib = calib.to(torch.float32).squeeze(0)
                img_out = visualize_objects_cv(_input, calib, tt_objects_torch)
                with lock:
                    shared_dict["output_image"] = img_out

            if not shared_dict["running"] or iter >= num_iter:
                shared_dict["running"] = False
                img_out = cv2.putText(img_out, "Press q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                with lock:
                    shared_dict["output_image"] = img_out
                break
        cv2_thread.join()

        pipeline.cleanup()


if __name__ == "__main__":
    args = parse_args()
    device = ttnn.CreateDevice(
        device_id=0,
        l1_small_size=32768,
        num_command_queues=2,
        trace_region_size=3000000,
    )
    from conftest import model_location_generator

    image_exts = {".jpg", ".jpeg", ".png"}
    if os.path.isdir(args.input):
        assert os.path.isdir(args.calib), "If --input is a directory, --calib must also be a directory."
        input_image_paths = []
        calib_paths = []
        calib_list = os.listdir(args.calib)
        for f in os.listdir(args.input):
            full_path = os.path.join(args.input, f)

            if os.path.isfile(full_path):
                basename, ext = os.path.splitext(f)
                ext = ext.lower()
                calib_candi_name = basename + ".txt"
                if ext in image_exts and calib_candi_name in calib_list:
                    input_image_paths.append(full_path)
                    calib_full_path = os.path.join(args.calib, calib_candi_name)
                    calib_paths.append(calib_full_path)
    else:
        input_image_paths = [args.input]
        calib_paths = [args.calib]

    main(device, model_location_generator, input_image_paths, calib_paths, args)
    ttnn.close_device(device)
