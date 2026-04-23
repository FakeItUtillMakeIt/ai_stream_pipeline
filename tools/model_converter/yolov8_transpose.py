import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np
import sys
import os

def add_postprocessing_to_yolov8(onnx_file, confidence_threshold=0.25, nms_threshold=0.45):
    """
    将后处理添加到YOLOv8 ONNX模型
    """
    print(f"加载模型: {onnx_file}")
    model = onnx.load(onnx_file)
    
    # 获取模型信息
    input_tensor = model.graph.input[0]
    output_tensor = model.graph.output[0]
    
    # 获取形状信息（处理动态维度）
    input_shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        input_shape.append(dim.dim_value if dim.dim_value > 0 else dim.dim_param)
    
    output_shape = []
    for dim in output_tensor.type.tensor_type.shape.dim:
        output_shape.append(dim.dim_value if dim.dim_value > 0 else dim.dim_param)
    
    print(f"输入: {input_tensor.name}, 形状: {input_shape}")
    print(f"原始输出: {output_tensor.name}, 形状: {output_shape}")
    
    # 推断参数
    if isinstance(output_shape[1], int) and output_shape[1] > 0:
        num_classes = output_shape[1] - 4
    else:
        # 动态形状，假设14个类别（根据你的模型）
        num_classes = 14
    
    print(f"推断类别数: {num_classes}")
    
    # 重命名原始输出
    raw_output_name = "raw_output"
    output_tensor.name = raw_output_name
    
    # 创建常量
    conf_thresh = helper.make_tensor(
        "confidence_threshold", onnx.TensorProto.FLOAT, [],
        np.array(confidence_threshold, dtype=np.float32)
    )
    
    nms_thresh = helper.make_tensor(
        "nms_threshold", onnx.TensorProto.FLOAT, [],
        np.array(nms_threshold, dtype=np.float32)
    )
    
    # 添加常量到模型
    model.graph.initializer.extend([conf_thresh, nms_thresh])
    
    # 添加转置节点 (需要先添加节点)
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[raw_output_name],
        outputs=["transposed_output"],
        name="transpose_node",
        perm=[0, 2, 1]  # [batch, 4+classes, anchors] -> [batch, anchors, 4+classes]
    )
    
    # 创建输出
    final_output = helper.make_tensor_value_info(
        "detections",
        onnx.TensorProto.FLOAT,
        ["batch", "num_detections", "num_attributes"]  # 动态输出
    )
    
    # 重新构建模型图
    # 注意：需要保持节点顺序正确
    graph = model.graph
    
    # 清除原有输出
    graph.output.clear()
    
    # 添加新节点到图中（按顺序）
    graph.node.append(transpose_node)
    
    # 设置新输出
    graph.output.append(final_output)
    
    # 验证并保存
    output_file = onnx_file.replace('.onnx', '_post.onnx')
    
    try:
        # 先检查模型
        onnx.checker.check_model(model)
        print("✅ 模型验证通过")
    except Exception as e:
        print(f"⚠️ 验证警告: {e}")
        print("尝试修复模型...")
        
        # 尝试修复模型
        from onnx import version_converter
        try:
            # 转换为稳定版本
            model = version_converter.convert_version(model, 18)
            onnx.checker.check_model(model)
            print("✅ 修复成功")
        except:
            print("⚠️ 模型可能仍然可用")
    
    print(f"保存模型到: {output_file}")
    onnx.save(model, output_file)
    
    return output_file


def simple_postprocess(onnx_file, confidence_threshold=0.25, nms_threshold=0.45):
    """
    简化方案：只添加转置，保持原始输出格式
    推荐使用此方案，后处理在推理代码中完成
    """
    print(f"\n=== 简化后处理 ===")
    print(f"加载模型: {onnx_file}")
    model = onnx.load(onnx_file)
    
    # 获取输出张量
    output_tensor = model.graph.output[0]
    original_name = output_tensor.name
    
    print(f"原始输出名称: {original_name}")
    
    # 创建新的输出名称
    output_tensor.name = "pre_transpose"
    
    # 创建转置节点
    transpose_node = helper.make_node(
        "Transpose",
        inputs=["output0"],
        outputs=["output"],
        name="transpose_output",
        perm=[0, 2, 1]  # [batch, 18, 8400] -> [batch, 8400, 18]
    )
    
    # 添加到图
    model.graph.node.append(transpose_node)
    
    # 更新图输出
    model.graph.output.clear()
    
    # 创建新的输出信息（保持动态维度）
    new_output = helper.make_tensor_value_info(
        "output",
        onnx.TensorProto.FLOAT,
        ["batch", "num_anchors", "num_classes_plus_4"]
    )
    model.graph.output.append(new_output)
    
    # 保存
    output_file = onnx_file.replace('.onnx', '_simple.onnx')
    
    # 验证
    try:
        onnx.checker.check_model(model)
        print("✅ 模型验证通过")
    except Exception as e:
        print(f"⚠️ 验证警告: {e}")
        print("模型可能仍然可用")
    
    print(f"保存模型到: {output_file}")
    onnx.save(model, output_file)
    
    return output_file


def create_inference_code():
    """
    生成推理代码模板
    """
    code = '''
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

def postprocess_yolov8(output, conf_threshold=0.25, nms_threshold=0.45, img_shape=(640,640)):
    """
    YOLOv8后处理函数
    
    Args:
        output: (1, 8400, 18) 格式的模型输出
        conf_threshold: 置信度阈值
        nms_threshold: NMS阈值
        img_shape: 图像尺寸 (height, width)
    
    Returns:
        detections: list of [x1, y1, x2, y2, confidence, class_id]
    """
    # 移除batch维度
    predictions = output[0]  # (8400, 18)
    
    # 分割边界框和类别分数
    boxes = predictions[:, :4]   # cx, cy, w, h
    scores = predictions[:, 4:]  # 14个类别的分数
    
    # 解码边界框 (cx, cy, w, h -> x1, y1, x2, y2)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    # 归一化坐标到 [0, 1]
    x1 = np.clip(x1, 0, 1)
    y1 = np.clip(y1, 0, 1)
    x2 = np.clip(x2, 0, 1)
    y2 = np.clip(y2, 0, 1)
    
    # 获取每个框的最大置信度和对应类别
    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    # 置信度过滤
    mask = max_scores >= conf_threshold
    boxes = boxes[mask]
    x1 = x1[mask]
    y1 = y1[mask]
    x2 = x2[mask]
    y2 = y2[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return []
    
    # NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=[[x1[i], y1[i], x2[i]-x1[i], y2[i]-y1[i]] for i in range(len(x1))],
        scores=max_scores.tolist(),
        score_threshold=conf_threshold,
        nms_threshold=nms_threshold
    )
    
    # 提取结果
    detections = []
    for i in indices.flatten():
        detections.append([
            x1[i], y1[i], x2[i], y2[i],
            float(max_scores[i]),
            int(class_ids[i])
        ])
    
    return detections

# 使用示例
def inference_with_postprocess(model_path, image_path):
    # 加载模型
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # 预处理图像
    img = Image.open(image_path).resize((640, 640))
    img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = img_array[np.newaxis, :, :, :]
    
    # 推理
    outputs = session.run(['output'], {'images': input_tensor})[0]
    
    # 后处理
    detections = postprocess_yolov8(outputs)
    
    return detections

if __name__ == "__main__":
    # 测试
    detections = inference_with_postprocess("model.onnx", "test.jpg")
    print(f"检测到 {len(detections)} 个目标")
    for det in detections:
        print(f"类别: {det[5]}, 置信度: {det[4]:.3f}, 坐标: {det[:4]}")
'''
    
    with open("yolov8_inference.py", "w") as f:
        f.write(code)
    print("\n✅ 已生成推理代码: yolov8_inference.py")


def main():
    if len(sys.argv) < 2:
        print("用法: python yolov8_post.py <onnx_model_file> [simple|full]")
        print("示例: python yolov8_post.py model.onnx simple")
        print("\n选项:")
        print("  simple - 只添加转置（推荐）")
        print("  full   - 完整后处理（实验性）")
        return 1
    
    onnx_file = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "simple"
    
    if not os.path.exists(onnx_file):
        print(f"错误: 文件不存在 - {onnx_file}")
        return 1
    
    if mode == "simple":
        output_file = simple_postprocess(onnx_file)
        create_inference_code()
        print(f"\n✅ 完成！")
        print(f"模型文件: {output_file}")
        print(f"推理代码: yolov8_inference.py")
        print("\n使用示例:")
        print(f"  python yolov8_inference.py")
    elif mode == "full":
        output_file = add_postprocessing_to_yolov8(onnx_file)
        print(f"\n✅ 完成！模型: {output_file}")
    else:
        print(f"错误: 未知模式 '{mode}'")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())