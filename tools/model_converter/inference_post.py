import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import sys

def inference_with_builtin_nms(model_path, image_path, conf_threshold=0.25, nms_threshold=0.45):
    """
    使用集成了NMS的ONNX模型进行推理
    """
    # 创建推理会话
    # 注意：如果使用自定义YOLOv8后处理，需要安装 onnxruntime-extensions
    providers = ['CPUExecutionProvider']
    
    # 尝试使用GPU
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("使用GPU推理")
    except:
        pass
    
    session = ort.InferenceSession(model_path, providers=providers)
    
    # 预处理图像
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    img_resized = img.resize((640, 640), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)[np.newaxis, :, :, :]
    
    # 推理
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    
    # 解析输出（根据你的模型输出格式）
    print(f"输出数量: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  输出{i}: shape={out.shape}, dtype={out.dtype}")
    
    # 解码检测结果
    detections = []
    
    # 如果输出是 selected_indices 格式
    if len(outputs) == 1 and outputs[0].shape[1] == 3:
        indices = outputs[0]  # [num_detections, 3]
        # 需要从原始输出中获取实际的边界框和分数
        print(f"检测到 {len(indices)} 个目标（需要进一步解码）")
    
    # 如果输出是 boxes, scores, class_ids 格式
    elif len(outputs) >= 3:
        boxes = outputs[0]  # [batch, num_detections, 4]
        scores = outputs[1]  # [batch, num_detections]
        class_ids = outputs[2]  # [batch, num_detections]
        
        # 缩放到原始图像尺寸
        scale_x = original_size[0] / 640
        scale_y = original_size[1] / 640
        
        for i in range(boxes.shape[1]):
            if scores[0, i] > 0:
                x1, y1, x2, y2 = boxes[0, i] * [scale_x, scale_y, scale_x, scale_y]
                detections.append([
                    float(x1), float(y1), float(x2), float(y2),
                    float(scores[0, i]), int(class_ids[0, i])
                ])
    
    return detections

def draw_detections(image_path, detections, output_path="result.jpg"):
    """绘制检测结果"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        color = colors[cls_id % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"class_{cls_id}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(output_path, img)
    print(f"结果保存到: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python inference_post.py model.onnx image.jpg")
        sys.exit(1)
    
    detections = inference_with_builtin_nms(sys.argv[1], sys.argv[2])
    print(f"检测到 {len(detections)} 个目标")
    draw_detections(sys.argv[2], detections)
