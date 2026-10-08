#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 ultralytics YOLOE 导出成 ai_stream_pipeline 现有 detection_infer 能直接吃的格式。

为什么需要这个脚本
------------------
仓库里现有的检测模型（如 models/sevncevision/*_nms.onnx）输出的是 5 个张量：

    det_boxes     (N, 4)   float32   xyxy，已在 640 输入坐标系内
    det_scores    (N,)     float32
    det_classes   (N, 1)   int64
    det_batch_ids (N, 1)   int64
    det_num_dets  ()       int64

而 ultralytics 的 `export(nms=True)` 给的是 (1, max_det, 4+nc+nm) 这种打包张量，
TensorRTDetectionEngine 那边是按 tensor 名字 setOutputTensor 取数据的
（见 src/nodes/infer/detection_infer.cpp: boxes_name_ = "det_boxes" 等），
所以直接塞 ultralytics 的输出会取不到数据。

这个脚本做两件事：
  1. 先导出 yoloe 的原生 end2end 图（含 NMS），拿到 xyxy 形式的框；
  2. 再套一层轻量 graph，把它的输出拆成上面 5 个张量，并按 ultralytics 约定
     补齐 metadata（names / stride / imgsz / task），使引擎能读到类名。

YOLOE 的关键约束
----------------
set_classes() 之后的文本嵌入已经烤进计算图，导出的 engine 不可再改类别。
因此类别顺序 == 输出通道顺序，改类别必须重新导出。
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import onnx

# 引擎期望的输出名，顺序与 detection_infer 的绑定一致
OUT_BOXES = "det_boxes"
OUT_SCORES = "det_scores"
OUT_CLASSES = "det_classes"
OUT_BATCH_IDS = "det_batch_ids"
OUT_NUM_DETS = "det_num_dets"


def export_raw(weight: str, classes, imgsz: int, max_det: int, device: str) -> str:
    """导出 ultralytics 原生 end2end ONNX，返回临时文件路径。"""
    from ultralytics import YOLOE

    model = YOLOE(weight)
    # 顺序即通道顺序，不可与检测时的 model_class_names 错位
    model.set_classes(list(classes))
    path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
        simplify=True,
        nms=True,
        batch=1,
        device=device,
        max_det=max_det,
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ultralytics 导出未产出文件: {path}")
    return path


def wrap_e2e(raw_path: str, classes, imgsz: int, out_path: str) -> None:
    """
    给 ultralytics 的 end2end 输出套一层 graph，拆成引擎需要的 5 个张量。

    ultralytics end2end 的 output0 布局是 (B, max_det, 4 + nc + nm)：
      [:, 0:4]  xyxy（已是像素坐标，未归一化）
      [:, 4:4+nc] 各类分数（已过 sigmoid/NMS）
    分割模型额外带 nm 个 mask 系数，与检测无关，这里直接丢掉。
    """
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(raw_path)
    graph = model.graph
    nc = len(classes)

    src = None
    for o in graph.output:
        if o.name == "output0":
            src = o
            break
    if src is None:
        raise RuntimeError(
            f"在 {raw_path} 里找不到 output0，实际输出: {[o.name for o in graph.output]}"
        )

    nc = len(classes)
    # 分割模型的 output0 最后一维 = 4 box + nc cls + nm mask系数（这里 38）。
    # 必须按真实维度做 Reshape，用 4+nc(=7) 去 reshape 38 维张量会形状不兼容。
    dims = [d.dim_value for d in src.type.tensor_type.shape.dim]
    actual_c = dims[-1] if dims and dims[-1] > 0 else 0
    max_det = dims[1] if len(dims) >= 2 and dims[1] > 0 else 0
    if max_det == 0:
        raise RuntimeError(f"output0 形状 {dims} 里拿不到静态的 max_det，无法固定 reshape")
    if actual_c and actual_c < 4 + nc:
        raise RuntimeError(
            f"output0 最后一维 {actual_c} 小于 4+{nc}，与类数不符，检查导出参数"
        )

    # batch 固定为 1：引擎按帧循环推理（detection_infer.cpp 的 batch_size 就是
    # 打包多少帧，不是真正的 batch 维），固定后形状完全确定，避免动态维歧义。

    total = 4 + nc

    # N 是"真实检出数"，逐帧变化。动态维必须用 dim_param，不能写 dim_value。
    boxes = helper.make_tensor_value_info(
        f"{OUT_BOXES}", TensorProto.FLOAT, ["num_dets", 4]
    )
    scores = helper.make_tensor_value_info(
        f"{OUT_SCORES}", TensorProto.FLOAT, ["num_dets"]
    )
    classes_out = helper.make_tensor_value_info(
        f"{OUT_CLASSES}", TensorProto.INT64, ["num_dets", 1]
    )
    batch_ids = helper.make_tensor_value_info(
        f"{OUT_BATCH_IDS}", TensorProto.INT64, ["num_dets", 1]
    )
    num_dets = helper.make_tensor_value_info(f"{OUT_NUM_DETS}", TensorProto.INT64, [])

    def const(name, arr):
        return helper.make_node(
            "Constant", [], [name],
            value=numpy_helper.from_array(arr, name=name + "_v"),
        )

    consts = [
        const("zero_i64", np.array(0, np.int64)),   # 0-D 标量，Range 要求
        const("neg_one_1", np.array([-1, 1], np.int64)),
        # [1, max_det, C] -> [max_det, C]。batch 维恒为 1，直接写死 max_det，
        # 不用 -1：reshape 后的第一维必须正好是 max_det，否则 Gather(axis=0)
        # 之后 boxes 会残留 batch 维变成 (N,1,4)，引擎按 (N,4) 读就错位。
        const("to_2d", np.array([max_det, actual_c], np.int64)),
        # Split 的 sizes 必须加和等于该维长度，所以要显式切出 mask 系数段
        # 再丢掉（引擎只要 box/scores/classes，不做分割）。
        const("split_sizes", np.array([4, nc, actual_c - 4 - nc], np.int64)),
        const("axis0_i", np.array([0], np.int64)),
        const("unsq_axis1", np.array([1], np.int64)),
        # Clamp 上界：NonZero 在空图上给空张量，夹到最后一行的合法下标
    ]

    nodes = consts + [
        # (B, max_det, C) -> (B*max_det, C)
        helper.make_node("Reshape", ["output0", "to_2d"], ["e2e_flat"]),
        # 最后一维切成 [box(4) | cls(nc) | mask]，mask 段丢弃
        helper.make_node("Split", ["e2e_flat", "split_sizes"],
                         [OUT_BOXES, "scores_flat", "_mask_dropped"], axis=1),
        # class = argmax(score)，reshape 成 (N,1)
        # 引擎按 scores[i] 一维消费 det_scores（detection_infer.cpp:567），
        # 所以 det_scores 必须是每行的最大分，而不是整张 (N,nc) 分数表。
        helper.make_node("ReduceMax", ["scores_flat"], ["best_score"],
                         axes=[1], keepdims=0),
        helper.make_node("Cast", ["best_score"], [OUT_SCORES], to=TensorProto.FLOAT),
        # class 用同一张 (N,nc) 分数表做 argmax
        helper.make_node("ArgMax", ["scores_flat"], ["cls_idx"], axis=1, keepdims=0),
        helper.make_node("Cast", ["cls_idx"], ["cls_flat"], to=TensorProto.INT64),
        helper.make_node("Reshape", ["cls_flat", "neg_one_1"], [OUT_CLASSES]),
        # ---- padding 行不压缩，交给引擎按分数过滤 ----
        #
        # 曾经试过 NonZero/Compress 把分数为 0 的行剔掉，但两种写法都过不去：
        #   Compress —— TensorRT 直接 "Failed to parse onnx file"（不支持该算子）
        #   NonZero —— 全 0 图上返回空 (0,2)，后续 Gather 越界
        #             ("idx=0 must be within the inclusive range [0,-1]")
        #
        # 其实不需要压缩。引擎按 det_num_dets 逐行读，再由
        # postprocessBatch 按 confidence_threshold 过滤（padding 行分数为 0，
        # 必然被阈值滤掉）。所以只要 batch_ids 合法、num_dets 真实即可。
        #
        # batch_ids 的语义是"这一行属于第几张图"，**不是行号**。
        # 之前误用 Range(0, max_det) 生成，运行期直接报
        # "Invalid batch_id 1 at det 1, max=1"。单 batch 图这里恒为 0。
        # batch_ids 全 0：整张图只有一个 batch。行数与 det_boxes 对齐。
        helper.make_node("Shape", [OUT_BOXES], ["nb_shape"]),
        helper.make_node("Gather", ["nb_shape", "axis0_i"], ["nb_1d"], axis=0),
        helper.make_node("Mul", ["nb_1d", "zero_i64"], ["batch_ids_row"]),
        # (N,) -> (N,1)：在第 1 维插入，不能用 axis=0
        helper.make_node("Unsqueeze", ["batch_ids_row", "unsq_axis1"], [OUT_BATCH_IDS]),
        # num_dets = det_boxes 的行数。padding 行也计数，交给引擎按分数过滤；
        # 少了这一项引擎会认为"没有输出"，或者读到不匹配的 buffer。
        helper.make_node("Identity", ["nb_1d"], [OUT_NUM_DETS]),
    ]

    graph.node.extend(nodes)

    del graph.output[:]
    graph.output.extend([boxes, scores, classes_out, batch_ids, num_dets])

    # metadata：引擎靠它读类名
    meta = {p.key: p.value for p in model.metadata_props}
    meta["names"] = str({i: c for i, c in enumerate(classes)})
    meta["stride"] = "32"
    meta["imgsz"] = str(imgsz)
    meta["task"] = "detect"
    meta["batch"] = "1"
    meta["channels"] = "3"
    meta["description"] = "YOLOE e2e wrapped for ai_stream_pipeline detection_infer"
    del model.metadata_props[:]
    for k, v in meta.items():
        p = model.metadata_props.add()
        p.key, p.value = k, str(v)

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, out_path)


def verify(out_path: str, imgsz: int, nc: int) -> None:
    """跑一张零图，确认输出名、dtype、形状都被引擎接受。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    expected = [OUT_BOXES, OUT_SCORES, OUT_CLASSES, OUT_BATCH_IDS, OUT_NUM_DETS]
    missing = [e for e in expected if e not in names]
    if missing:
        raise RuntimeError(f"缺少引擎需要的输出: {missing}（实际 {names}）")

    feed = {sess.get_inputs()[0].name: np.zeros((1, 3, imgsz, imgsz), np.float32)}
    out = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, feed))}
    print(f"  输出: {[(n, out[n].shape, out[n].dtype.name) for n in expected]}")
    if out[OUT_NUM_DETS].size and int(out[OUT_NUM_DETS]) > 0:
        n = int(np.ravel(out[OUT_NUM_DETS])[0])
        if out[OUT_CLASSES].size:
            assert int(out[OUT_CLASSES].max()) < nc, "class id 越界"
        print(f"  检出 {n} 个目标，首个 box(xyxy)={out[OUT_BOXES][0].round(1)}")
    print("  metadata names:", {p.key: p.value for p in
                                onnx.load(out_path, load_external_data=False)
                                .metadata_props}["names"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="yoloe-v8s-seg.pt")
    ap.add_argument("--classes", nargs="+", default=["child", "wall", "gate"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = "/tmp/_yoloe_raw_end2end.onnx"
    print(f"[1/3] 导出原生 end2end（类别 {args.classes}，顺序即通道顺序）")
    raw = export_raw(args.weight, args.classes, args.imgsz, args.max_det, args.device)
    shutil.copyfile(raw, tmp)

    print("[2/3] 包装成 det_boxes/det_scores/det_classes/det_batch_ids/det_num_dets")
    wrap_e2e(tmp, args.classes, args.imgsz, args.out)

    print("[3/3] 校验")
    verify(args.out, args.imgsz, len(args.classes))

    # 类名 sidecar：配置里引它，避免手抄导致通道错位
    with open(os.path.splitext(args.out)[0] + ".names.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "names": list(args.classes),
                "note": "通道顺序 == 此顺序；detector_config.model_class_names 必须一致",
                "source_weight": os.path.basename(args.weight),
                "prompt_note": "set_classes 后文本嵌入已烤入图，engine 不可改类别",
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    print(f"完成: {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())