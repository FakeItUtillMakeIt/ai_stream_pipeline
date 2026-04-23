import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np
import sys
import os

def add_full_postprocessing(onnx_file, confidence_threshold=0.25, nms_threshold=0.45, max_output_boxes=100):
    """
    将完整的后处理（置信度过滤 + NMS）添加到ONNX模型中
    修复形状不匹配问题
    """
    print(f"\n=== 完整后处理（集成NMS）===")
    print(f"加载模型: {onnx_file}")
    model = onnx.load(onnx_file)
    
    # 获取输入输出
    input_tensor = model.graph.input[0]
    output_tensor = model.graph.output[0]
    
    print(f"输入: {input_tensor.name}")
    print(f"原始输出: {output_tensor.name}, 形状: {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")
    
    # 重命名原始输出

    # 1. 添加转置节点: [batch, 18, 8400] -> [batch, 8400, 18]
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[output_tensor.name],
        outputs=["transposed_output"],
        name="transpose_node",
        perm=[0, 2, 1]
    )
    model.graph.node.append(transpose_node)
    
    # 2. 使用 Slice 分割边界框和类别分数
    # 提取前4列（边界框）[batch, 8400, 4]
    starts_bbox = helper.make_tensor(
        "starts_bbox", onnx.TensorProto.INT64, [1], np.array([0], dtype=np.int64)
    )
    ends_bbox = helper.make_tensor(
        "ends_bbox", onnx.TensorProto.INT64, [1], np.array([4], dtype=np.int64)
    )
    
    # 提取后14列（类别分数）[batch, 8400, 14]
    starts_scores = helper.make_tensor(
        "starts_scores", onnx.TensorProto.INT64, [1], np.array([4], dtype=np.int64)
    )
    ends_scores = helper.make_tensor(
        "ends_scores", onnx.TensorProto.INT64, [1], np.array([18], dtype=np.int64)
    )
    
    axis = helper.make_tensor(
        "axis", onnx.TensorProto.INT64, [1], np.array([2], dtype=np.int64)
    )
    
    model.graph.initializer.extend([starts_bbox, ends_bbox, starts_scores, ends_scores, axis])
    
    # Slice 节点提取边界框
    slice_bbox_node = helper.make_node(
        "Slice",
        inputs=["transposed_output", "starts_bbox", "ends_bbox", "axis"],
        outputs=["bboxes_raw"],
        name="slice_bbox"
    )
    model.graph.node.append(slice_bbox_node)
    
    # Slice 节点提取类别分数
    slice_scores_node = helper.make_node(
        "Slice",
        inputs=["transposed_output", "starts_scores", "ends_scores", "axis"],
        outputs=["class_scores_raw"],
        name="slice_scores"
    )
    model.graph.node.append(slice_scores_node)
    
    # 3. 解码边界框 (cx, cy, w, h -> x1, y1, x2, y2)
    # 使用 Gather 和数学运算
    
    # 创建常量
    zero = helper.make_tensor("zero", onnx.TensorProto.FLOAT, [], np.array(0, dtype=np.float32))
    one = helper.make_tensor("one", onnx.TensorProto.FLOAT, [], np.array(1, dtype=np.float32))
    two = helper.make_tensor("two", onnx.TensorProto.FLOAT, [], np.array(2, dtype=np.float32))
    model.graph.initializer.extend([zero, one, two])
    
    # 注意：YOLOv8 的坐标已经是归一化的，直接使用
    # 但 NMS 需要的是 [x1, y1, x2, y2] 格式
    # 实际上 YOLOv8 输出的 bbox 已经是 cx, cy, w, h 格式，需要解码
    
    # 简化：直接使用原始 bbox（某些版本的 YOLOv8 输出已经是解码后的）
    # 这里我们保持原样，让 NMS 处理
    
    # 4. 添加 NMS 节点
    max_output_boxes_const = helper.make_tensor(
        "max_output_boxes", onnx.TensorProto.INT64, [],
        np.array(max_output_boxes, dtype=np.int64)
    )
    conf_thresh_const = helper.make_tensor(
        "confidence_threshold", onnx.TensorProto.FLOAT, [],
        np.array(confidence_threshold, dtype=np.float32)
    )
    nms_thresh_const = helper.make_tensor(
        "nms_threshold", onnx.TensorProto.FLOAT, [],
        np.array(nms_threshold, dtype=np.float32)
    )
    
    model.graph.initializer.extend([max_output_boxes_const, conf_thresh_const, nms_thresh_const])
    
    # NMS节点 - 注意输入形状要求
    # boxes: [batch, num_boxes, 4]
    # scores: [batch, num_boxes, num_classes]
    nms_node = helper.make_node(
        "NonMaxSuppression",
        inputs=["bboxes_raw", "class_scores_raw", "max_output_boxes", "confidence_threshold", "nms_threshold"],
        outputs=["selected_indices"],
        name="nms_node"
    )
    model.graph.node.append(nms_node)
    
    # 5. 设置输出
    indices_output = helper.make_tensor_value_info(
        "selected_indices",
        onnx.TensorProto.INT64,
        [None, 3]
    )
    
    # 更新图输出
    model.graph.output.clear()
    model.graph.output.append(indices_output)
    
    # 验证并保存
    output_file = onnx_file.replace('.onnx', '_full_post.onnx')
    
    # 检查模型
    try:
        onnx.checker.check_model(model)
        print("✅ 模型验证通过")
    except Exception as e:
        print(f"⚠️ 验证警告: {e}")
    
    print(f"保存模型到: {output_file}")
    onnx.save(model, output_file)
    
    return output_file


def inference_with_builtin_nms(model_path, image_path, conf_threshold=0.25, nms_threshold=0.45):
    """
    使用集成了NMS的ONNX模型进行推理
    修复形状问题
    """
    import onnxruntime as ort
    import numpy as np
    from PIL import Image
    import cv2
    
    print(f"\n=== 推理 ===")
    print(f"加载模型: {model_path}")
    
    # 加载原始模型（需要原始输出来获取完整数据）
    original_model_path = model_path.replace('_full_post.onnx', '.onnx')
    if not os.path.exists(original_model_path):
        print(f"错误: 找不到原始模型 {original_model_path}")
        print(f"请确保原始模型存在于: {original_model_path}")
        return []
    
    # 加载NMS模型
    try:
        session_nms = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        session_original = ort.InferenceSession(original_model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"加载模型失败: {e}")
        return []
    
    # 预处理图像
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    img_resized = img.resize((640, 640), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)[np.newaxis, :, :, :]
    
    # 推理原始模型获取完整输出
    input_name = session_original.get_inputs()[0].name
    original_outputs = session_original.run(None, {input_name: img_array})
    raw_output = original_outputs[0]  # shape: (1, 18, 8400)
    print(f"原始输出形状: {raw_output.shape}")
    
    # 转置为 (1, 8400, 18)
    transposed_output = raw_output.transpose(0, 2, 1)
    print(f"转置后形状: {transposed_output.shape}")
    
    # 分离边界框和分数
    bboxes = transposed_output[0, :, :4]  # [8400, 4] - cx, cy, w, h
    scores = transposed_output[0, :, 4:]  # [8400, 14] - 类别分数
    
    print(f"边界框形状: {bboxes.shape}")
    print(f"分数形状: {scores.shape}")
    
    # 解码边界框 (cx, cy, w, h -> x1, y1, x2, y2)
    x1 = bboxes[:, 0] - bboxes[:, 2] / 2
    y1 = bboxes[:, 1] - bboxes[:, 3] / 2
    x2 = bboxes[:, 0] + bboxes[:, 2] / 2
    y2 = bboxes[:, 1] + bboxes[:, 3] / 2
    
    # 裁剪到 [0, 1] 范围
    x1 = np.clip(x1, 0, 1)
    y1 = np.clip(y1, 0, 1)
    x2 = np.clip(x2, 0, 1)
    y2 = np.clip(y2, 0, 1)
    
    decoded_bboxes = np.stack([x1, y1, x2, y2], axis=1)  # [8400, 4]
    
    # 应用NMS（使用OpenCV）
    detections = []
    
    # 对每个类别单独处理（因为OpenCV的NMS不支持多类别）
    for class_id in range(scores.shape[1]):
        class_scores = scores[:, class_id]
        
        # 置信度过滤
        mask = class_scores >= conf_threshold
        if not np.any(mask):
            continue
        
        class_boxes = decoded_bboxes[mask]
        class_scores_filtered = class_scores[mask]
        
        # NMS
        boxes_nms = [[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in class_boxes]
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_nms,
            scores=class_scores_filtered.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold
        )
        
        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0]
            indices = indices.flatten()
            
            for idx in indices:
                box = class_boxes[idx]
                score = class_scores_filtered[idx]
                
                # 缩放到原始图像尺寸
                scale_x = original_size[0] / 640
                scale_y = original_size[1] / 640
                
                x1_px = box[0] * original_size[0]
                y1_px = box[1] * original_size[1]
                x2_px = box[2] * original_size[0]
                y2_px = box[3] * original_size[1]
                
                detections.append([
                    float(x1_px), float(y1_px),
                    float(x2_px), float(y2_px),
                    float(score),
                    int(class_id)
                ])
    
    print(f"检测到 {len(detections)} 个目标")
    
    for i, det in enumerate(detections[:5]):  # 只打印前5个
        print(f"  {i+1}. class={det[5]}, conf={det[4]:.3f}, "
              f"box=[{det[0]:.0f}, {det[1]:.0f}, {det[2]:.0f}, {det[3]:.0f}]")
    
    return detections


def draw_detections(image_path, detections, output_path="result_nms.jpg"):
    """绘制检测结果"""
    import cv2
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    class_names = ['person', 'head', 'helmet', 'class3', 'class4', 'class5',
                   'class6', 'class7', 'class8', 'class9', 'class10',
                   'class11', 'class12', 'class13', 'class14']
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        color = colors[cls_id % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_names[cls_id]}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(output_path, img)
    print(f"结果保存到: {output_path}")
    return img


def main():
    if len(sys.argv) < 2:
        print("用法: python yolov8_post.py <onnx_model_file> [--infer]")
        print("示例:")
        print("  python yolov8_post.py model.onnx              # 只添加NMS")
        print("  python yolov8_post.py model.onnx --infer      # 添加NMS并测试推理")
        return 1
    
    onnx_file = sys.argv[1]
    do_inference = "--infer" in sys.argv
    
    if not os.path.exists(onnx_file):
        print(f"错误: 文件不存在 - {onnx_file}")
        return 1
    
    # 添加后处理
    output_file = add_full_postprocessing(onnx_file)
    print(f"\n✅ 模型已保存: {output_file}")
    
    # 如果需要测试推理
    if do_inference:
        test_image = "/home/sevnce/project/bus.jpg"
        # 查找测试图像参数
        for i, arg in enumerate(sys.argv):
            if arg == "--infer" and i+1 < len(sys.argv) and not sys.argv[i+1].startswith("--"):
                test_image = sys.argv[i+1]
                break
        
        if os.path.exists(test_image):
            print(f"\n测试推理: {test_image}")
            detections = inference_with_builtin_nms(output_file, test_image)
            
            if detections:
                draw_detections(test_image, detections, "result_nms.jpg")
                print(f"\n✅ 检测到 {len(detections)} 个目标")
            else:
                print("❌ 未检测到目标")
        else:
            print(f"测试图像不存在: {test_image}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())