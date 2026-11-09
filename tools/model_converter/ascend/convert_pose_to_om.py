#!/usr/bin/env python3
"""YOLO-Pose ONNX -> Ascend .om 转换

原始 yolov8n_pose.onnx 是 opset 18 且带动态 shape；ATC 的 ONNX 插件
(libops_all_onnx_plugin.so) 在展开 Resize/Split 时会段错误。这里的处理：

1. onnxsim 把输入固定成 [1,3,640,640]，输出变成 [1,8400,56]；
2. Resize 补一个 ROI 常量输入，并去掉 opset 18 才有的
   antialias / keep_aspect_ratio_policy 属性；
3. Split 的 sizes 输入转成 opset 13 的 split 属性，去掉 num_outputs；
4. opset 设为 13；
5. 调 ATC 生成 .om。

推荐输入 `models/yolov8n/yolov8n_pose.trans.onnx`（输出 [1,8400,56]）。
`yolov8n_pose.onnx`（输出 [1,56,8400]）也能转，AscendPoseEstimation
会自动检测并做转置，但 trans 版本路径更稳定。

用法（先 source CANN set_env.sh）：
  python3 tools/model_converter/ascend/convert_pose_to_om.py \
      models/yolov8n/yolov8n_pose.trans.onnx \
      --output models/yolov8n/yolov8n_pose

依赖：onnx、onnxsim、numpy
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    import onnxsim
except ImportError as e:
    print("需要 onnx / onnxsim / numpy：python3 -m pip install --user onnx onnxsim numpy",
          file=sys.stderr)
    raise


def _strip_opset18_attrs(model):
    """Resize 去掉 opset 18 属性；Split 的 sizes 输入转 attr。"""
    consts = {}
    for t in model.graph.initializer:
        consts[t.name] = numpy_helper.to_array(t)
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)

    new_nodes = []
    roi_idx = 0
    for n in model.graph.node:
        if n.op_type == "Resize":
            keep = [a for a in n.attribute
                    if a.name not in ("antialias", "keep_aspect_ratio_policy")]
            del n.attribute[:]
            n.attribute.extend(keep)
            # opset 13 Resize 允许空 roi，但部分 ATC 版本对空输入处理异常；
            # 显式给一个全 0 的 roi 常量更稳。
            if len(n.input) >= 3 and n.input[1] == "":
                name = f"resize_roi_{roi_idx}"
                roi_idx += 1
                roi = helper.make_tensor(name, onnx.TensorProto.FLOAT, [8],
                                         np.zeros(8, dtype=np.float32))
                new_nodes.append(helper.make_node("Constant", [], [name], value=roi))
                n.input[1] = name
        elif n.op_type == "Split":
            sizes = None
            if len(n.input) > 1 and n.input[1] in consts:
                sizes = consts[n.input[1]].astype(np.int64).tolist()
                del n.input[1:]
            attrs = [a for a in n.attribute if a.name != "num_outputs"]
            del n.attribute[:]
            n.attribute.extend(attrs)
            if sizes is not None:
                n.attribute.extend([helper.make_attribute("split", sizes)])
        new_nodes.append(n)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    for o in model.opset_import:
        if o.domain == "":
            o.version = 13
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="输入 YOLO-Pose ONNX（推荐 *_pose.trans.onnx）")
    ap.add_argument("--output", default=None,
                    help="输出前缀，默认与输入同目录同名")
    ap.add_argument("--size", type=int, default=640, help="模型输入边长，默认 640")
    ap.add_argument("--soc_version", default="Ascend310P3")
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    out_prefix = args.output or os.path.splitext(args.onnx)[0]
    if out_prefix.endswith(".om"):
        out_prefix = out_prefix[:-3]

    print(f"[pose] load {args.onnx}")
    model = onnx.load(args.onnx, load_external_data=False)

    print(f"[pose] onnxsim (input {args.size}x{args.size}) ...")
    sm, ok = onnxsim.simplify(
        model, overwrite_input_shapes={"images": [1, 3, args.size, args.size]})
    if not ok:
        print("[pose] onnxsim check failed", file=sys.stderr)
        return 1
    sm = _strip_opset18_attrs(sm)

    tmpdir = tempfile.mkdtemp(prefix="yolo_pose_")
    prepared = os.path.join(tmpdir, "prepared.onnx")
    onnx.save(sm, prepared)
    print(f"[pose] prepared: {prepared}")
    for i in sm.graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param
                for d in i.type.tensor_type.shape.dim]
        print(f"[pose]   in  {i.name} {dims}")
    for o in sm.graph.output:
        dims = [d.dim_value if d.dim_value else d.dim_param
                for d in o.type.tensor_type.shape.dim]
        print(f"[pose]   out {o.name} {dims}")

    atc = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0") + "/bin/atc"
    if not os.path.exists(atc):
        atc = "atc"
    cmd = [
        atc, "--framework=5", f"--model={prepared}",
        f"--input_shape=images:1,3,{args.size},{args.size}",
        "--input_format=NCHW", f"--soc_version={args.soc_version}",
        f"--output={out_prefix}", "--log=error",
    ]
    print("[pose] ATC:", " ".join(cmd))

    env = os.environ.copy()
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0")
    ascend_py = os.path.join(ascend_home, "python", "site-packages")
    parts = [p for p in env.get("PYTHONPATH", "").split(":") if p]
    if ascend_py not in parts:
        parts.insert(0, ascend_py)
    env["PYTHONPATH"] = ":".join(parts)
    ld_path = env.get("LD_LIBRARY_PATH", "")
    for d in (os.path.join(ascend_home, "lib64"),
              "/usr/local/Ascend/driver/lib64",
              "/usr/local/Ascend/driver/lib64/common",
              "/usr/local/Ascend/driver/lib64/driver"):
        if d not in ld_path.split(":") and os.path.isdir(d):
            ld_path = d + (":" + ld_path if ld_path else "")
    env["LD_LIBRARY_PATH"] = ld_path

    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(f"[pose] ATC failed with code {res.returncode}", file=sys.stderr)
        return 1

    produced = out_prefix + ".om"
    if not os.path.exists(produced):
        candidates = glob.glob(out_prefix + "*.om")
        if candidates:
            os.replace(candidates[0], produced)
    if not os.path.exists(produced):
        print(f"[pose] 未找到生成的 .om（检查 {out_prefix}*.om）", file=sys.stderr)
        return 1
    print(f"[pose] done: {produced}")

    if not args.keep_temp:
        try:
            os.remove(prepared)
            os.rmdir(tmpdir)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
