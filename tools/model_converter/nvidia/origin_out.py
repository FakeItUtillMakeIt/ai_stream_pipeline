import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("/home/sevnce/project/ai_stream_pipeline/models/sevncevision/20260422_sevnce_14cls_nms.onnx")  # 原始模型
inp = sess.get_inputs()[0]
print(f"输入: {inp.name}, shape={inp.shape}")

out = sess.get_outputs()[0]
print(f"输出: {out.name}, shape={out.shape}")

# 跑一个随机输入看实际输出形状
dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
result = sess.run(None, {inp.name: dummy})
print(f"实际输出形状: {result[0].shape}")
print(f"输出数值范围: [{result[0].min():.3f}, {result[0].max():.3f}]")
print(f"输出前5个样本:\n{result[0][0, :5, :]}")