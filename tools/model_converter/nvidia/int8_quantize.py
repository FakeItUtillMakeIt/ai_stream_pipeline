#!/usr/bin/env python3
"""
INT8 模型量化工具
用于将 FP16/FP32 TensorRT 模型转换为 INT8 量化模型

使用方法:
    python int8_quantize.py \
        --onnx model.onnx \
        --output model_int8.engine \
        --calibration_data ./calibration_images/ \
        --batch_size 4 \
        --num_calibration_batches 32

边缘部署加速效果:
    - 模型大小减少 4x (FP32 → INT8)
    - 推理速度提升 2-4x
    - 功耗降低 30-50%
"""

import argparse
import os
import sys
import glob
from pathlib import Path

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    import numpy as np
    import cv2
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Install with: pip install tensorrt pycuda numpy opencv-python")
    sys.exit(1)


class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Entropy Calibrator for TensorRT"""

    def __init__(self, calibration_data_path, batch_size, input_shape, input_name, cache_file):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape  # (C, H, W)
        self.input_name = input_name
        self.cache_file = cache_file

        # 收集校准图片
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        if os.path.isdir(calibration_data_path):
            self.image_paths = [
                str(p) for p in Path(calibration_data_path).rglob('*')
                if p.suffix.lower() in image_extensions
            ]
        else:
            self.image_paths = [calibration_data_path]

        if len(self.image_paths) < batch_size:
            print(f"Warning: Only {len(self.image_paths)} images, need at least {batch_size}")

        print(f"[Calibrator] Found {len(self.image_paths)} calibration images")
        print(f"[Calibrator] Batch size: {batch_size}, Input shape: {input_shape}")

        self.current_index = 0
        self.total_batches = len(self.image_paths) // batch_size

        # 分配 GPU 内存
        self.input_size = batch_size * np.prod(input_shape)
        self.d_input = cuda.mem_alloc(self.input_size * np.float32().nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.total_batches:
            return None

        batch_data = []
        start_idx = self.current_index * self.batch_size

        for i in range(self.batch_size):
            img_path = self.image_paths[start_idx + i]
            img = self._preprocess_image(img_path)
            batch_data.append(img)

        batch = np.stack(batch_data, axis=0).astype(np.float32)

        cuda.memcpy_htod(self.d_input, batch)

        self.current_index += 1
        if self.current_index % 10 == 0:
            print(f"[Calibrator] Progress: {self.current_index}/{self.total_batches} batches")

        return [self.d_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[Calibrator] Reading calibration cache: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"[Calibrator] Wrote calibration cache: {self.cache_file} ({len(cache)} bytes)")

    def _preprocess_image(self, image_path):
        """Preprocess image: resize + normalize + HWC→NCHW"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Failed to read {image_path}, using black image")
            img = np.zeros((self.input_shape[1], self.input_shape[2], 3), dtype=np.uint8)

        # Resize
        img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # HWC → NCHW (BGR → RGB)
        img = img[:, :, ::-1].transpose(2, 0, 1)

        return img


def build_int8_engine(onnx_path, output_path, calibration_data,
                      batch_size=1, num_calibration_batches=32,
                      input_shape=(3, 640, 640), input_name="images",
                      workspace_size=4):
    """
    Build INT8 TensorRT engine from ONNX model

    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save INT8 engine
        calibration_data: Path to calibration images (directory or file)
        batch_size: Calibration batch size
        num_calibration_batches: Number of calibration batches
        input_shape: Input tensor shape (C, H, W)
        input_name: Input tensor name
        workspace_size: Workspace size in GB
    """
    print(f"\n{'='*60}")
    print(f"INT8 Model Quantization Tool")
    print(f"{'='*60}")
    print(f"ONNX Model:          {onnx_path}")
    print(f"Output Engine:       {output_path}")
    print(f"Calibration Data:    {calibration_data}")
    print(f"Batch Size:          {batch_size}")
    print(f"Calibration Batches: {num_calibration_batches}")
    print(f"Input Shape:         {input_shape}")
    print(f"Workspace Size:      {workspace_size} GB")
    print(f"{'='*60}\n")

    # Check files
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found: {onnx_path}")
        return False

    if not os.path.exists(calibration_data):
        print(f"Error: Calibration data not found: {calibration_data}")
        return False

    # Create logger
    logger = trt.Logger(trt.Logger.INFO)

    # Create builder
    builder = trt.Builder(logger)

    # Create network
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # Create parser
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    print(f"[1/5] Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Error: Failed to parse ONNX model")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            return False

    print(f"  Network inputs: {network.num_inputs}")
    print(f"  Network outputs: {network.num_outputs}")

    # Create config
    print(f"[2/5] Configuring INT8 quantization")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # Fallback to FP16
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))

    # Create calibrator
    cache_file = output_path.replace('.engine', '.calib')
    calibrator = Int8EntropyCalibrator(
        calibration_data,
        batch_size,
        input_shape,
        input_name,
        cache_file
    )
    config.int8_calibrator = calibrator

    # Build engine
    print(f"[3/5] Building INT8 engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Error: Failed to build engine")
        return False

    # Save engine
    print(f"[4/5] Saving engine to: {output_path}")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(serialized_engine)

    # Report
    engine_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[5/5] Done!")
    print(f"\n{'='*60}")
    print(f"INT8 Engine Built Successfully!")
    print(f"{'='*60}")
    print(f"Engine Size:     {engine_size_mb:.2f} MB")
    print(f"Engine Path:     {output_path}")
    print(f"Cache Path:      {cache_file}")
    print(f"\nPerformance Benefits:")
    print(f"  - Model size:    4x smaller than FP32")
    print(f"  - Inference:     2-4x faster")
    print(f"  - Power:         30-50% lower")
    print(f"{'='*60}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='INT8 Model Quantization Tool for Edge Deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic quantization
  python int8_quantize.py \\
      --onnx yolov8n.onnx \\
      --output yolov8n_int8.engine \\
      --calibration_data ./calibration_images/

  # Custom batch size and calibration batches
  python int8_quantize.py \\
      --onnx model.onnx \\
      --output model_int8.engine \\
      --calibration_data ./images/ \\
      --batch_size 4 \\
      --num_calibration_batches 64

Edge Deployment Benefits:
  - 4x smaller model (FP32 → INT8)
  - 2-4x faster inference
  - 30-50% lower power consumption
        """
    )

    parser.add_argument('--onnx', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save INT8 engine')
    parser.add_argument('--calibration_data', type=str, required=True,
                        help='Path to calibration images (directory or file)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Calibration batch size (default: 1)')
    parser.add_argument('--num_calibration_batches', type=int, default=32,
                        help='Number of calibration batches (default: 32)')
    parser.add_argument('--input_shape', type=str, default='3,640,640',
                        help='Input shape C,H,W (default: 3,640,640)')
    parser.add_argument('--input_name', type=str, default='images',
                        help='Input tensor name (default: images)')
    parser.add_argument('--workspace', type=int, default=4,
                        help='Workspace size in GB (default: 4)')

    args = parser.parse_args()

    # Parse input shape
    input_shape = tuple(int(x) for x in args.input_shape.split(','))
    if len(input_shape) != 3:
        print(f"Error: Input shape must be C,H,W, got {args.input_shape}")
        sys.exit(1)

    success = build_int8_engine(
        onnx_path=args.onnx,
        output_path=args.output,
        calibration_data=args.calibration_data,
        batch_size=args.batch_size,
        num_calibration_batches=args.num_calibration_batches,
        input_shape=input_shape,
        input_name=args.input_name,
        workspace_size=args.workspace
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
