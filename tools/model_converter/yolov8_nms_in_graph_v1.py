import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import sys
import argparse
import onnx
from onnx import helper, TensorProto


# ========================== 1. 模型转换：修复 spatial_dimension 不匹配 ==========================

def add_nms_to_yolov8(onnx_path, output_path, num_classes=14,
                      max_detections=300, iou_threshold=0.45, score_threshold=0.25):
    model = onnx.load(onnx_path)
    graph = model.graph

    opset_version = model.opset_import[0].version if model.opset_import else 13
    print(f"原模型 opset: {opset_version}")

    if len(graph.output) != 1:
        print(f"⚠️ 警告: 模型有 {len(graph.output)} 个输出")
    orig_output = graph.output[0].name

    node_idx = [0]
    def next_name(prefix):
        node_idx[0] += 1
        return f"{prefix}_{node_idx[0]}"

    def const(name, dtype, shape, vals):
        return helper.make_tensor(name, dtype, shape, vals)

    nc = 4 + num_classes

    # ---------- 1. Slice boxes [1, 4, 8400] ----------
    slice_boxes = helper.make_node(
        'Slice',
        inputs=[orig_output, 'starts_0', 'ends_4', 'axis_1', 'steps_1'],
        outputs=['boxes_sliced'],
        name=next_name('slice_boxes'),
    )

    # ---------- 2. Transpose boxes -> [1, 8400, 4] ----------
    transpose_boxes = helper.make_node(
        'Transpose',
        inputs=['boxes_sliced'],
        outputs=['boxes_raw'],
        perm=[0, 2, 1],
        name=next_name('transpose_boxes'),
    )

    # ---------- 3. Slice scores [1, 14, 8400] ----------
    slice_scores = helper.make_node(
        'Slice',
        inputs=[orig_output, 'starts_4', 'ends_nc', 'axis_1', 'steps_1'],
        outputs=['scores_raw'],
        name=next_name('slice_scores'),
    )

    # ---------- 4. Sigmoid ----------
    sigmoid_scores = helper.make_node(
        'Sigmoid',
        inputs=['scores_raw'],
        outputs=['scores'],
        name=next_name('sigmoid'),
    )

    # ---------- 5. NMS ----------
    max_out = const('max_out', TensorProto.INT64, [1], [max_detections])
    iou_thres = const('iou_thres', TensorProto.FLOAT, [1], [iou_threshold])
    score_thres = const('score_thres', TensorProto.FLOAT, [1], [score_threshold])

    nms_node = helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes_raw', 'scores', 'max_out', 'iou_thres', 'score_thres'],
        outputs=['selected_indices'],
        center_point_box=1,
        name=next_name('nms'),
    )

    # ---------- 6. 提取索引列 ----------
    slice_batch = helper.make_node(
        'Slice',
        inputs=['selected_indices', 'starts_0_', 'ends_1_', 'axis_1_', 'steps_1'],
        outputs=['batch_col'],
        name=next_name('slice_batch'),
    )
    slice_class = helper.make_node(
        'Slice',
        inputs=['selected_indices', 'starts_1_', 'ends_2_', 'axis_1_', 'steps_1'],
        outputs=['det_classes'],
        name=next_name('slice_class'),
    )
    slice_box = helper.make_node(
        'Slice',
        inputs=['selected_indices', 'starts_2_', 'ends_3_', 'axis_1_', 'steps_1'],
        outputs=['box_col'],
        name=next_name('slice_box'),
    )

    concat_indices = helper.make_node(
        'Concat',
        inputs=['batch_col', 'box_col'],
        outputs=['gather_boxes_indices'],
        axis=1,
        name=next_name('concat'),
    )

    # ---------- 7. GatherND ----------
    gather_boxes = helper.make_node(
        'GatherND',
        inputs=['boxes_raw', 'gather_boxes_indices'],
        outputs=['det_boxes'],
        name=next_name('gather_boxes'),
    )
    gather_scores = helper.make_node(
        'GatherND',
        inputs=['scores', 'selected_indices'],
        outputs=['det_scores'],
        name=next_name('gather_scores'),
    )

    # ---------- 8. 计算检测数（简化版）----------
    shape_indices = helper.make_node(
        'Shape',
        inputs=['selected_indices'],
        outputs=['indices_shape'],
        name=next_name('shape_indices')
    )
    gather_num_dets = helper.make_node(
        'Gather',
        inputs=['indices_shape', 'dim_0'],
        outputs=['num_dets_1d'],
        axis=0,
        name=next_name('gather_num_dets')
    )
    squeeze_num_dets = helper.make_node(
        'Squeeze',
        inputs=['num_dets_1d', 'axes_sq'],
        outputs=['det_num_dets'],
        name=next_name('squeeze_num_dets')
    )

    # ---------- 9. 更新输出 ----------
    graph.output.clear()
    graph.output.extend([
        helper.make_tensor_value_info('det_boxes', TensorProto.FLOAT, [None, 4]),
        helper.make_tensor_value_info('det_scores', TensorProto.FLOAT, [None]),
        helper.make_tensor_value_info('det_classes', TensorProto.INT64, [None, 1]),
        helper.make_tensor_value_info('det_num_dets', TensorProto.INT64, []),  # 标量
    ])

    # ---------- 10. Initializer ----------
    init_list = [
        const('starts_0', TensorProto.INT64, [1], [0]),
        const('ends_4', TensorProto.INT64, [1], [4]),
        const('starts_4', TensorProto.INT64, [1], [4]),
        const('ends_nc', TensorProto.INT64, [1], [nc]),
        const('axis_1', TensorProto.INT64, [1], [1]),
        const('steps_1', TensorProto.INT64, [1], [1]),
        max_out, iou_thres, score_thres,
        const('starts_0_', TensorProto.INT64, [1], [0]),
        const('ends_1_', TensorProto.INT64, [1], [1]),
        const('starts_1_', TensorProto.INT64, [1], [1]),
        const('ends_2_', TensorProto.INT64, [1], [2]),
        const('starts_2_', TensorProto.INT64, [1], [2]),
        const('ends_3_', TensorProto.INT64, [1], [3]),
        const('axis_1_', TensorProto.INT64, [1], [1]),
        const('dim_0', TensorProto.INT64, [1], [0]),
        const('axes_sq', TensorProto.INT64, [1], [0]),
    ]
    graph.initializer.extend(init_list)

    # ---------- 11. 节点 ----------
    graph.node.extend([
        slice_boxes, transpose_boxes, slice_scores, sigmoid_scores,
        nms_node,
        slice_batch, slice_class, slice_box, concat_indices,
        gather_boxes, gather_scores,
        shape_indices, gather_num_dets, squeeze_num_dets,
    ])

    # ---------- 12. 保存 ----------
    try:
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, output_path)
        print(f"✅ 成功保存: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        onnx.save(model, output_path + '.debug.onnx')
        return False

# ========================== 2. 推理代码 ==========================

def preprocess(image_path, input_size=640):
    original_image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = original_image.size
    img = original_image.resize((input_size, input_size), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    tensor = arr[np.newaxis, :, :, :]
    return tensor, (orig_w, orig_h)


def is_nms_model(session):
    names = [o.name for o in session.get_outputs()]
    return 'det_boxes' in names and 'det_scores' in names


def inference_with_nms(session, image_path, input_size=640):
    input_name = session.get_inputs()[0].name
    input_tensor, (orig_w, orig_h) = preprocess(image_path, input_size)

    # 现在有4个输出
    det_boxes, det_scores, det_classes, det_num_dets = session.run(None, {input_name: input_tensor})
    
    num_dets = int(det_num_dets)  # 标量
    print(f"检测数: {num_dets}")

    detections = []
    for i in range(num_dets):  # 只遍历实际检测数
        cx, cy, w, h = det_boxes[i]
        x1 = max(0, cx - w / 2)
        y1 = max(0, cy - h / 2)
        x2 = min(input_size, cx + w / 2)
        y2 = min(input_size, cy + h / 2)

        cls_id = int(det_classes[i][0]) if det_classes.ndim > 1 else int(det_classes[i])

        detections.append([
            float(x1), float(y1), float(x2), float(y2),
            float(det_scores[i]), cls_id
        ])

    return detections, (orig_w, orig_h)

def postprocess_yolov8(output, conf_threshold=0.25, nms_threshold=0.45, img_size=640):
    """原始YOLOv8后处理（适配 (1, 18, 8400) 格式）"""
    predictions = output[0]  # (18, 8400)
    predictions = predictions.transpose(1, 0)  # (8400, 18)

    boxes = predictions[:, :4]
    scores = predictions[:, 4:]

    scores = 1.0 / (1.0 + np.exp(-scores))

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    x1 = np.clip(x1, 0, img_size)
    y1 = np.clip(y1, 0, img_size)
    x2 = np.clip(x2, 0, img_size)
    y2 = np.clip(y2, 0, img_size)

    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

    mask = max_scores >= conf_threshold
    if not np.any(mask):
        return []

    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    max_scores, class_ids = max_scores[mask], class_ids[mask]

    valid = (x2 > x1) & (y2 > y1)
    if not np.any(valid):
        return []

    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    max_scores, class_ids = max_scores[valid], class_ids[valid]

    boxes_nms = [[float(x1[i]), float(y1[i]), float(x2[i]-x1[i]), float(y2[i]-y1[i])]
                 for i in range(len(x1))]
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_nms,
        scores=max_scores.tolist(),
        score_threshold=conf_threshold,
        nms_threshold=nms_threshold
    )

    detections = []
    if len(indices) > 0:
        indices = indices.flatten() if not isinstance(indices, tuple) else indices[0].flatten()
        for i in indices:
            detections.append([
                float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                float(max_scores[i]), int(class_ids[i])
            ])
    return detections


def inference_without_nms(session, image_path, conf_threshold=0.25, nms_threshold=0.45):
    input_name = session.get_inputs()[0].name
    input_tensor, (orig_w, orig_h) = preprocess(image_path)

    outputs = session.run(None, {input_name: input_tensor})
    output = outputs[0]

    detections = postprocess_yolov8(
        output,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        img_size=640
    )
    return detections, (orig_w, orig_h)


def draw_detections(image_path, detections, class_names=None, output_path="result.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return None

    img_h, img_w = img.shape[:2]
    scale_x, scale_y = img_w / 640, img_h / 640

    colors = [
        (255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
        (0,255,255),(128,0,0),(0,128,0),(0,0,128),(128,128,0),
        (128,0,128),(0,128,128),(255,128,0),(128,255,0),(0,128,255)
    ]

    if class_names is None:
        class_names = [f'class_{i}' for i in range(15)]

    for det in detections:
        x1_m, y1_m, x2_m, y2_m, conf, cls_id = det
        x1 = int(x1_m * scale_x)
        y1 = int(y1_m * scale_y)
        x2 = int(x2_m * scale_x)
        y2 = int(y2_m * scale_y)

        if x1 >= x2 or y1 >= y2:
            continue

        color = colors[cls_id % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
        label = f"{name}: {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        ly = max(y1 - 5, th + 5)
        cv2.rectangle(img, (x1, ly - th - baseline - 5), (x1 + tw, ly), color, -1)
        cv2.putText(img, label, (x1, ly - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    info = f"Detected: {len(detections)} objects"
    cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"结果已保存: {output_path}")
    return img


# ========================== 3. 主函数 ==========================

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 ONNX 推理（支持内置NMS模型）')
    parser.add_argument('model_path', help='ONNX模型路径')
    parser.add_argument('image_path', nargs='?', help='输入图像路径（转换模式可不填）')
    parser.add_argument('--convert', action='store_true', help='将原始模型转换为带NMS的模型')
    parser.add_argument('--output-model', default='model_nms.onnx', help='转换后模型保存路径')
    parser.add_argument('--num-classes', type=int, default=14, help='类别数量（默认14）')
    parser.add_argument('--max-dets', type=int, default=300, help='每类最大检测数（默认300）')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--nms', type=float, default=0.45, help='NMS阈值（仅对原始模型有效）')
    parser.add_argument('--output', default='result.jpg', help='输出图像路径')

    args = parser.parse_args()

    if args.convert:
        success = add_nms_to_yolov8(
            args.model_path,
            args.output_model,
            num_classes=args.num_classes,
            max_detections=args.max_dets,
            iou_threshold=args.nms,
            score_threshold=args.conf,
        )
        if success:
            print(f"\n💡 转换完成，请使用新模型推理: python {sys.argv[0]} {args.output_model} <image.jpg>")
        return 0

    if not args.image_path:
        parser.error("推理模式需要提供 image_path")

    print(f"加载模型: {args.model_path}")
    session = ort.InferenceSession(args.model_path, providers=['CPUExecutionProvider'])

    if is_nms_model(session):
        print("🔍 检测到模型已内置NMS，直接输出过滤结果")
        detections, _ = inference_with_nms(session, args.image_path)
    else:
        print("🔍 检测到原始模型，使用Python后处理")
        detections, _ = inference_without_nms(
            session, args.image_path,
            conf_threshold=args.conf,
            nms_threshold=args.nms
        )

    print(f"\n检测到 {len(detections)} 个目标:")
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls_id = det
        print(f"  {i+1}. 类别:{cls_id}, 置信度:{conf:.3f}, "
              f"坐标:({x1:.0f},{y1:.0f})->({x2:.0f},{y2:.0f})")

    class_names = ['person', 'head', 'helmet', 'class3', 'class4', 'class5',
                   'class6', 'class7', 'class8', 'class9', 'class10',
                   'class11', 'class12', 'class13', 'class14']

    if detections:
        draw_detections(args.image_path, detections, class_names, args.output)
    else:
        img = cv2.imread(args.image_path)
        cv2.putText(img, "No objects detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(args.output, img)
        print(f"已保存无检测结果图片: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())