#!/usr/bin/env python3
"""RelateAnything ONNX -> Ascend .om 转换（含 If/RoPE 静态化）

原始 relateanything.onnx (opset 17) 包含 9 个 If 子图，且子图引用外层
RoPE 的 Shape/Range。ATC 能编译，但运行期 GE 的 shape inference 会报
"[1,1568,0] cannot broadcast"。这里按当前导出的语义把 If 内联成静态分支，
再用 onnxsim 固定 image/boxes/box_counts/W/alpha 的 shape，最后调用 ATC。

用法（先 source CANN set_env.sh）：
  python3 tools/model_converter/ascend/convert_relation_to_om.py \
      models/relateanything/relateanything.onnx \
      --output models/relateanything/relateanything.om

依赖：onnx、onnxsim（pip install --user onnx onnxsim）
"""
import argparse
import copy
import glob
import os
import subprocess
import sys
import tempfile

try:
    import onnx
    from onnx import helper
    import onnxsim
except ImportError as e:
    print("需要 onnx / onnxsim：python3 -m pip install --user onnx onnxsim", file=sys.stderr)
    raise

# 当前 relateanything.onnx 导出的 If 分支选择：
# - RoPE 的第一个 If 走 then_branch（计算位置编码）
# - RoPE 的 If_1 条件 Less(rank-1, 0) 为 false，走 else_branch（Identity）
# - 其余 attention 的 If 条件为 batch/dim==1，走 then_branch（Squeeze）
BRANCH_CHOICES = {
    "/model/model/rope_embeddings/If": "then_branch",
    "/model/model/rope_embeddings/If_1": "else_branch",
    "/model/spatial_pool/cross_attn/If": "then_branch",
    "/model/spatial_pool/cross_attn_1/If": "then_branch",
    "/model/rel_transformer/cross_layers.0/multihead_attn/If": "then_branch",
    "/model/rel_transformer/last_cross/cross_attn/If": "then_branch",
    "/model/rel_interaction/dep_layers.0/self_attn/If": "then_branch",
    "/model/rel_interaction/dep_layers.1/self_attn/If": "then_branch",
    "/model/rel_interaction/gnd_layers.0/self_attn/If": "then_branch",
}

INPUT_SHAPES = {
    "image": [1, 3, 448, 448],
    "boxes": [1, 32, 4],
    "box_counts": [1],
    "W": [243, 512],
    "alpha": [243],
}


def inline_ifs(model):
    new_nodes = []
    idx = 0
    replaced = 0
    for n in model.graph.node:
        if n.op_type != "If":
            new_nodes.append(n)
            continue
        choice = BRANCH_CHOICES.get(n.name, "then_branch")
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        g = attrs[choice]
        prefix = f"inl{idx}_"
        idx += 1
        out_map = {o.name: prefix + o.name for o in g.output}
        for bn in g.node:
            nn = copy.deepcopy(bn)
            nn.name = prefix + bn.name
            nn.input[:] = [out_map.get(x, x) for x in bn.input]
            nn.output[:] = [out_map.get(x, x) for x in bn.output]
            new_nodes.append(nn)
        new_nodes.append(helper.make_node(
            "Identity", [out_map[g.output[0].name]], [n.output[0]], name=prefix + "if_out"))
        replaced += 1
        print(f"[relation] inline If {n.name} -> {choice} ({len(g.node)} nodes)")
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return replaced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="输入 relateanything.onnx")
    ap.add_argument("--output", default=None,
                    help="输出前缀/ .om 路径，默认同目录 relateanything")
    ap.add_argument("--soc_version", default="Ascend310P3")
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    out_prefix = args.output or os.path.splitext(args.onnx)[0]
    if out_prefix.endswith(".om"):
        out_prefix = out_prefix[:-3]

    print(f"[relation] load {args.onnx}")
    model = onnx.load(args.onnx, load_external_data=False)

    replaced = inline_ifs(model)
    if replaced == 0:
        print("[relation] 警告：没有找到 If 节点，模型可能不是当前导出格式", file=sys.stderr)
    onnx.checker.check_model(model)

    tmpdir = tempfile.mkdtemp(prefix="relateanything_")
    inlined = os.path.join(tmpdir, "inlined.onnx")
    simplified = os.path.join(tmpdir, "simplified.onnx")
    onnx.save(model, inlined)

    print("[relation] onnxsim ...")
    sm, ok = onnxsim.simplify(model, overwrite_input_shapes=INPUT_SHAPES)
    if not ok:
        print("[relation] onnxsim check failed", file=sys.stderr)
        return 1
    onnx.save(sm, simplified)

    atc = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0") + "/bin/atc"
    if not os.path.exists(atc):
        atc = "atc"
    shape_arg = ";".join(f"{k}:{','.join(map(str, v))}" for k, v in INPUT_SHAPES.items())
    cmd = [
        atc, "--framework=5", f"--model={simplified}",
        f"--input_shape={shape_arg}", f"--soc_version={args.soc_version}",
        f"--output={out_prefix}", "--log=error",
    ]
    print("[relation] ATC:", " ".join(cmd))
    # 用户可能把 onnx/onnxsim 装在自定义目录并通过 PYTHONPATH 注入；
    # 这里确保 ATC 子进程仍能看到 CANN 的 python/site-packages（tbe 等）。
    env = os.environ.copy()
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0")
    ascend_py = os.path.join(ascend_home, "python", "site-packages")
    py_path = env.get("PYTHONPATH", "")
    parts = [p for p in py_path.split(":") if p]
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
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(f"[relation] ATC failed with code {res.returncode}", file=sys.stderr)
        return 1

    produced = out_prefix + ".om"
    if not os.path.exists(produced):
        candidates = glob.glob(out_prefix + "*.om")
        if candidates:
            os.replace(candidates[0], produced)
    if not os.path.exists(produced):
        print(f"[relation] 未找到生成的 .om（检查 {out_prefix}*.om）", file=sys.stderr)
        return 1
    print(f"[relation] done: {produced}")
    if not args.keep_temp:
        for p in (inlined, simplified):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
