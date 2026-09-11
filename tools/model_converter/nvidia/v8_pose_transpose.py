import onnx
from onnx import helper, TensorProto

def transpose_pose_output(onnx_path, output_path):
    """
    在 YOLOv8-pose ONNX 模型末尾插入 Transpose 节点，
    将输出从 [batch, 56, N] 转换为 [batch, N, 56]
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    # 获取原输出信息
    original_output = graph.output[0]
    orig_name = original_output.name
    orig_shape = [d.dim_value if d.dim_value > 0 else None for d in original_output.type.tensor_type.shape.dim]

    print(f"原输出名称: {orig_name}")
    print(f"原输出形状: {orig_shape}")

    # 创建 Transpose 节点: perm=[0, 2, 1]
    # 输入: 原输出名  输出: 新名字
    transposed_name = orig_name + "_transposed"
    transpose_node = helper.make_node(
        'Transpose',
        inputs=[orig_name],
        outputs=[transposed_name],
        perm=[0, 2, 1],  # [batch, 56, N] -> [batch, N, 56]
        name='Transpose_output'
    )

    # 将节点添加到图中
    graph.node.append(transpose_node)

    # 更新 graph 的输出定义
    # 新形状: [batch, N, 56]，其中 N 和 batch 保持动态/原样
    new_shape_dims = []
    if len(orig_shape) >= 3:
        # batch
        new_shape_dims.append(original_output.type.tensor_type.shape.dim[0])
        # N (原第2维)
        new_shape_dims.append(original_output.type.tensor_type.shape.dim[2])
        # 56 (原第1维)
        new_shape_dims.append(original_output.type.tensor_type.shape.dim[1])
    else:
        # 如果形状信息不完整，直接创建动态维度
        new_shape_dims = [
            helper.make_tensor_value_info('', TensorProto.UNDEFINED, []).type.tensor_type.shape.dim[0],
            helper.make_tensor_value_info('', TensorProto.UNDEFINED, []).type.tensor_type.shape.dim[0],
            helper.make_tensor_value_info('', TensorProto.UNDEFINED, []).type.tensor_type.shape.dim[0],
        ]

    new_output = helper.make_tensor_value_info(
        transposed_name,
        original_output.type.tensor_type.elem_type,
        []  # 形状在后面单独设置
    )
    # 手动复制维度信息
    new_output.type.tensor_type.shape.dim.extend(new_shape_dims)

    # 替换输出
    graph.output.clear()
    graph.output.append(new_output)

    # 保存
    try:
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, output_path)
        print(f"✅ 转置完成，已保存: {output_path}")
        print(f"   新输出名称: {transposed_name}")
        print(f"   新输出形状: [batch, N, 56]")
        return True
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        onnx.save(model, output_path + '.debug.onnx')
        return False


# ========== 使用示例 ==========
if __name__ == "__main__":
    transpose_pose_output("/home/sevnce/project/ai_stream_pipeline/models/yolov8n/yolov8n_pose.onnx", "/home/sevnce/project/ai_stream_pipeline/models/yolov8n/yolov8n_pose.trans.onnx")