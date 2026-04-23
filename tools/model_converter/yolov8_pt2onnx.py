from ultralytics import YOLO
import argparse

def model_convert(file):
    
    model = YOLO(file)
    model.export(format='onnx',dynamic=True,opset=12)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_path', type=str, required=True)
    args = parser.parse_args()
    model_convert(args.pt_path)
