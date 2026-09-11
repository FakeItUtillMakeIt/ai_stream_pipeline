import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import sys
import argparse

def postprocess_yolov8(output, conf_threshold=0.25, nms_threshold=0.45, img_size=640):
    """
    YOLOv8后处理函数
    
    Args:
        output: 模型输出 (1, 8400, 18)
        conf_threshold: 置信度阈值
        nms_threshold: NMS阈值
        img_size: 模型输入图像尺寸（默认640）
    """
    # 移除batch维度
    predictions = output[0]  # (8400, 18)
    
    # 分割边界框和类别分数
    boxes = predictions[:, :4]   # cx, cy, w, h (像素坐标)
    scores = predictions[:, 4:]  # 类别分数
    
    print(f"原始box统计: min={boxes.min():.2f}, max={boxes.max():.2f}, mean={boxes.mean():.2f}")
    
    # 解码边界框 (cx, cy, w, h -> x1, y1, x2, y2)
    # 注意：坐标是像素坐标，范围在 [0, img_size] 内
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    # 裁剪到图像范围内 [0, img_size]
    x1 = np.clip(x1, 0, img_size)
    y1 = np.clip(y1, 0, img_size)
    x2 = np.clip(x2, 0, img_size)
    y2 = np.clip(y2, 0, img_size)
    
    print(f"解码后坐标: x1范围=[{x1.min():.2f}, {x1.max():.2f}], y1范围=[{y1.min():.2f}, {y1.max():.2f}]")
    
    # 获取每个框的最大置信度和对应类别
    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    print(f"置信度统计: min={max_scores.min():.4f}, max={max_scores.max():.4f}, mean={max_scores.mean():.4f}")
    print(f"置信度 >= {conf_threshold} 的框数量: {np.sum(max_scores >= conf_threshold)}")
    
    # 置信度过滤
    mask = max_scores >= conf_threshold
    if not np.any(mask):
        return []
    
    x1 = x1[mask]
    y1 = y1[mask]
    x2 = x2[mask]
    y2 = y2[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    
    # 过滤掉无效的边界框（宽度或高度为0）
    valid_mask = (x2 > x1) & (y2 > y1)
    if not np.any(valid_mask):
        print("警告：所有边界框都无效")
        return []
    
    x1 = x1[valid_mask]
    y1 = y1[valid_mask]
    x2 = x2[valid_mask]
    y2 = y2[valid_mask]
    max_scores = max_scores[valid_mask]
    class_ids = class_ids[valid_mask]
    
    # NMS（使用像素坐标）
    boxes_nms = [[x1[i], y1[i], x2[i] - x1[i], y2[i] - y1[i]] for i in range(len(x1))]
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_nms,
        scores=max_scores.tolist(),
        score_threshold=conf_threshold,
        nms_threshold=nms_threshold
    )
    
    # 提取结果
    detections = []
    if len(indices) > 0:
        if isinstance(indices, tuple):
            indices = indices[0]
        indices = indices.flatten()
        
        for i in indices:
            detections.append([
                float(x1[i]),
                float(y1[i]),
                float(x2[i]),
                float(y2[i]),
                float(max_scores[i]),
                int(class_ids[i])
            ])
            
            print(f"检测到: 类别={class_ids[i]}, 置信度={max_scores[i]:.3f}, "
                  f"坐标=[{x1[i]:.0f}, {y1[i]:.0f}, {x2[i]:.0f}, {y2[i]:.0f}]")
    
    return detections

def draw_detections(image_path, detections, class_names=None, output_path="result.jpg"):
    """
    在图像上绘制检测结果
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return None
    
    img_height, img_width = img.shape[:2]
    print(f"图像尺寸: {img_width}x{img_height}")
    
    # 定义颜色
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (128, 255, 0), (0, 128, 255)
    ]
    
    # 默认类别名称
    if class_names is None:
        class_names = [f'class_{i}' for i in range(15)]
    
    # 计算缩放比例（因为模型输入是640x640，原始图像可能不同）
    # 注意：模型输出的坐标是基于640x640的，需要缩放到原始图像尺寸
    model_size = 640
    scale_x = img_width / model_size
    scale_y = img_height / model_size
    
    print(f"缩放比例: x={scale_x:.3f}, y={scale_y:.3f}")
    
    # 绘制每个检测结果
    for det in detections:
        x1_model, y1_model, x2_model, y2_model, conf, cls_id = det
        
        # 将模型坐标缩放到原始图像坐标
        x1_px = int(x1_model * scale_x)
        y1_px = int(y1_model * scale_y)
        x2_px = int(x2_model * scale_x)
        y2_px = int(y2_model * scale_y)
        
        print(f"绘制: 类别={cls_id}, 置信度={conf:.3f}, "
              f"模型坐标=({x1_model:.0f},{y1_model:.0f})->({x2_model:.0f},{y2_model:.0f}), "
              f"像素坐标=({x1_px},{y1_px})->({x2_px},{y2_px})")
        
        # 确保坐标有效
        if x1_px >= x2_px or y1_px >= y2_px:
            print(f"警告: 无效的边界框坐标，跳过")
            continue
        
        # 获取颜色
        color = colors[cls_id % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(img, (x1_px, y1_px), (x2_px, y2_px), color, 2)
        
        # 准备标签文本
        class_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
        label = f"{class_name}: {conf:.2f}"
        
        # 计算文本大小
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # 绘制标签背景
        label_y = max(y1_px - 5, label_height + 5)
        cv2.rectangle(
            img,
            (x1_px, label_y - label_height - baseline - 5),
            (x1_px + label_width, label_y),
            color,
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            img,
            label,
            (x1_px, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    
    # 显示检测统计
    info_text = f"Detected: {len(detections)} objects"
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    # 保存结果
    cv2.imwrite(output_path, img)
    print(f"结果已保存到: {output_path}")
    
    return img

def inference_and_visualize(model_path, image_path, conf_threshold=0.25, nms_threshold=0.45, output_path="result.jpg"):
    """
    完整的推理和可视化流程
    """
    print(f"加载模型: {model_path}")
    
    # 加载ONNX模型
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"输入节点: {input_name}")
    print(f"输出节点: {output_name}")
    
    # 读取并预处理图像
    print(f"处理图像: {image_path}")
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size  # (width, height)
    print(f"原始图像尺寸: {original_size}")
    
    # 调整到模型输入尺寸 (640x640)
    input_image = original_image.resize((640, 640), Image.Resampling.LANCZOS)
    
    # 转换为numpy数组并归一化
    input_array = np.array(input_image).astype(np.float32) / 255.0
    
    # 转换为CHW格式 (C, H, W)
    input_array = input_array.transpose(2, 0, 1)
    
    # 添加batch维度 (1, C, H, W)
    input_tensor = input_array[np.newaxis, :, :, :]
    
    print(f"输入形状: {input_tensor.shape}")
    
    # 推理
    print("运行推理...")
    outputs = session.run([output_name], {input_name: input_tensor})
    output = outputs[0]
    print(f"输出形状: {output.shape}")
    
    # 后处理（使用640作为模型输入尺寸）
    print("后处理...")
    detections = postprocess_yolov8(
        output, 
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        img_size=640  # 模型输入尺寸
    )
    
    # 打印检测结果
    print(f"\n检测到 {len(detections)} 个目标:")
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls_id = det
        print(f"  {i+1}. 类别: {cls_id}, 置信度: {conf:.3f}, "
              f"模型坐标: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
    
    # 绘制结果
    if detections:
        class_names = ['person', 'head', 'helmet', 'class3', 'class4', 'class5',
                      'class6', 'class7', 'class8', 'class9', 'class10',
                      'class11', 'class12', 'class13', 'class14']
        
        draw_detections(image_path, detections, class_names, output_path)
    else:
        print("未检测到任何目标")
        img = cv2.imread(image_path)
        cv2.putText(img, "No objects detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(output_path, img)
        print(f"已保存无检测结果图片: {output_path}")
    
    return detections

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 ONNX 推理和可视化')
    parser.add_argument('model_path', help='ONNX模型路径')
    parser.add_argument('image_path', help='输入图像路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值 (默认: 0.25)')
    parser.add_argument('--nms', type=float, default=0.45, help='NMS阈值 (默认: 0.45)')
    parser.add_argument('--output', default='result.jpg', help='输出图像路径 (默认: result.jpg)')
    
    args = parser.parse_args()
    
    # 运行推理和可视化
    detections = inference_and_visualize(
        args.model_path,
        args.image_path,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        output_path=args.output
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())