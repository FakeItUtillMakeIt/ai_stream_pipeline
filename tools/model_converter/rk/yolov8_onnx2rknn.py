import argparse
from rknn.api import RKNN


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX model to RKNN format')
    parser.add_argument('--onnx', type=str, required=True,
                        help='Path to the input ONNX model')
    parser.add_argument('--rknn', type=str, required=True,
                        help='Path to save the output RKNN model')
    parser.add_argument('--target', type=str, default='rk3588',
                        help='Target platform (default: rk3588)')
    parser.add_argument('--quant', action='store_true',
                        help='Enable quantization (default: False)')
    parser.add_argument('--mean', type=float, nargs=3, default=[0, 0, 0],
                        help='Mean values for normalization (default: 0 0 0)')
    parser.add_argument('--std', type=float, nargs=3, default=[255, 255, 255],
                        help='Std values for normalization (default: 255 255 255)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset file for quantization (required if --quant is set)')
    parser.add_argument('--input-name', type=str, default='images',
                        help='Name of the input tensor (default: images)')
    parser.add_argument('--input-size', type=int, nargs=4, default=[1, 3, 640, 640],
                        help='Input size list, e.g. --input-size 1 3 640 640')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quant and args.dataset is None:
        print('Error: --dataset is required when --quant is enabled.')
        exit(1)

    rknn = RKNN(verbose=True)

    rknn.config(mean_values=[args.mean],
                std_values=[args.std],
                target_platform=args.target)

    print(f'Loading ONNX model: {args.onnx}')
    ret = rknn.load_onnx(
        model=args.onnx,
        inputs=[args.input_name],
        input_size_list=[args.input_size]
    )
    if ret != 0:
        print('Load model failed!')
        exit(ret)

    print(f'Building model (quantization={args.quant})...')
    ret = rknn.build(do_quantization=args.quant, dataset=args.dataset)
    if ret != 0:
        print('Build model failed!')
        exit(ret)

    print(f'Exporting RKNN model to: {args.rknn}')
    ret = rknn.export_rknn(args.rknn)
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)

    print(f'RKNN model saved to: {args.rknn}')
    rknn.release()


if __name__ == '__main__':
    main()