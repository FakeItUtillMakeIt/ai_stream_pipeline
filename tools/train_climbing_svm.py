#!/usr/bin/env python3
"""
训练攀爬检测 SVM / XGBoost 模型

用法:
    # 用现有规则引擎自动打标（CSV里有label列）
    python train_climbing_svm.py --csv training_data.csv --output svm_model.csv

    # 训练XGBoost并导出ONNX（可选）
    python train_climbing_svm.py --csv training_data.csv --output svm_model.csv --use-xgb --onnx xgb_model.onnx

    # 检查数据质量，看正负样本分布和特征统计
    python train_climbing_svm.py --csv training_data.csv --inspect

CSV格式（由climbing_detector导出）:
    sample_id,label,hand_above_shoulder,arm_bend,knee_raise,center_raise,
    body_tilt,limb_span,alternating_limb,overall_ascent,has_ascent,
    oscillation,lateral_movement,movement_burst,net_displacement,ascent_slope
"""

import argparse
import csv
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json


FEATURE_NAMES = [
    "hand_above_shoulder",    # 0
    "arm_bend",               # 1
    "knee_raise",             # 2
    "center_raise",           # 3
    "body_tilt",              # 4
    "limb_span",              # 5
    "alternating_limb",       # 6
    "overall_ascent",         # 7
    "has_ascent",             # 8
    "oscillation",            # 9
    "lateral_movement",       # 10
    "movement_burst",         # 11
    "net_displacement_ratio", # 12
    "ascent_slope_ratio",     # 13
]
FEATURE_DIM = len(FEATURE_NAMES)


def load_csv(path):
    data, labels, meta = [], [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat = [float(row[name]) for name in FEATURE_NAMES]
            data.append(feat)
            labels.append(int(row['label']))
            meta.append({
                'video_id': row.get('video_id', ''),
                'frame_id': row.get('frame_id', ''),
                'track_id': row.get('track_id', ''),
                'sample_id': row.get('sample_id', ''),
            })
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return data, labels, meta


def inspect(data, labels, meta):
    pos = data[labels == 1]
    neg = data[labels == 0]

    videos = set(m['video_id'] for m in meta)
    print(f"总样本数: {len(labels)}")
    print(f"视频源数: {len(videos)}")
    print(f"正样本 (攀爬): {len(pos)} ({100*len(pos)/len(labels):.1f}%)")
    print(f"负样本 (其他): {len(neg)} ({100*len(neg)/len(labels):.1f}%)")
    print()

    if len(videos) > 1:
        for vid in sorted(videos):
            vmask = np.array([m['video_id'] == vid for m in meta])
            vpos = labels[vmask].sum()
            print(f"  {vid}: {vmask.sum()} 样本, {vpos} 正 ({100*vpos/vmask.sum():.1f}%)")
        print()
    print(f"{'特征':<22} {'正样本均值':>10} {'负样本均值':>10} {'差异':>10}")
    print("-" * 52)
    for i, name in enumerate(FEATURE_NAMES):
        p = pos[:, i].mean()
        n = neg[:, i].mean()
        diff = p - n
        print(f"{name:<22} {p:>10.4f} {n:>10.4f} {diff:>10.4f}")


def export_svm_model(model, scaler, path):
    coef = model.coef_.flatten()
    intercept = model.intercept_[0]
    with open(path, 'w') as f:
        f.write(f"dim,{FEATURE_DIM},0\n")
        for i, w in enumerate(coef):
            f.write(f"weight,{i},{w:.6f}\n")
        f.write(f"bias,0,{intercept:.6f}\n")
    print(f"模型导出至: {path}")


def export_scaler(scaler, path):
    mean = scaler.mean_.tolist()
    scale = scaler.scale_.tolist()
    with open(path, 'w') as f:
        json.dump({"mean": mean, "scale": scale}, f)
    print(f"归一化参数导出至: {path}")


def train_svm(data, labels):
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    model = LinearSVC(C=1.0, class_weight='balanced', max_iter=10000, dual=False)

    # 交叉验证
    scores = cross_val_score(model, X, labels, cv=5, scoring='f1')
    print(f"5折交叉验证F1: {scores.mean():.4f} (±{scores.std():.4f})")
    print()

    # 训练集/测试集评估
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("测试集评估:")
    print(classification_report(y_test, y_pred, target_names=["其他", "攀爬"], zero_division=0))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    print()
    print("特征权重 (SVM coef绝对值):")
    importances = np.abs(model.coef_.flatten())
    indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices):
        print(f"  {rank+1}. {FEATURE_NAMES[idx]:<22} |coef| = {importances[idx]:.6f}")

    return model, scaler


def train_xgboost(data, labels):
    try:
        import xgboost as xgb
    except ImportError:
        print("需要安装 xgboost: pip install xgboost")
        return None, None

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    pos_weight = (labels == 0).sum() / max((labels == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("XGBoost 测试集评估:")
    print(classification_report(y_test, y_pred, target_names=["其他", "攀爬"], zero_division=0))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    print()
    print("XGBoost 特征重要性:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices):
        print(f"  {rank+1}. {FEATURE_NAMES[idx]:<22} importance = {importances[idx]:.6f}")

    # 导出为 ONNX（可选，之后可用ONNX Runtime在C++推理）
    try:
        import onnxmltools
        from onnxconverter_common.data_types import FloatTensorType
    except ImportError:
        print("跳过ONNX导出: pip install onnxmltools skl2onnx onnxconverter_common")
        return model, scaler

    initial_type = [('float_input', FloatTensorType([None, FEATURE_DIM]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
    return model, scaler, onnx_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='训练数据CSV路径')
    parser.add_argument('--output', type=str, default='svm_model.csv', help='SVM模型输出路径')
    parser.add_argument('--scaler-out', type=str, default='scaler.json', help='归一化参数输出路径')
    parser.add_argument('--inspect', action='store_true', help='仅检查数据质量')
    parser.add_argument('--use-xgb', action='store_true', help='使用XGBoost代替SVM')
    parser.add_argument('--onnx', type=str, help='XGBoost ONNX模型路径')
    parser.add_argument('--svm-kernel', type=str, default='linear', choices=['linear', 'rbf', 'poly'],
                        help='SVM核函数 (仅linear可导出到C++)')
    parser.add_argument('--balanced', action='store_true', default=True, help='使用class_weight=balanced')
    parser.add_argument('--downsample', type=float, default=0.0,
                        help='负样本下采样比例 (如0.3=保留30%负样本)')
    args = parser.parse_args()

    data, labels, meta = load_csv(args.csv)
    print(f"加载 {len(data)} 条样本，{FEATURE_DIM} 维特征")

    if args.inspect:
        inspect(data, labels, meta)
        return

    if args.downsample > 0:
        pos_mask = labels == 1
        neg_mask = labels == 0
        neg_indices = np.where(neg_mask)[0]
        keep_n = int(len(neg_indices) * args.downsample)
        keep_neg = np.random.choice(neg_indices, size=keep_n, replace=False)
        keep_mask = np.zeros(len(labels), dtype=bool)
        keep_mask[pos_mask] = True
        keep_mask[keep_neg] = True
        data = data[keep_mask]
        labels = labels[keep_mask]
        print(f"下采样后: {len(data)} 条样本")
        inspect(data, labels, meta)
        print()

    if args.use_xgb:
        result = train_xgboost(data, labels)
        if result is None:
            return
        if len(result) == 2:
            model, scaler = result
        else:
            model, scaler, onnx_model = result
            if onnx_model and args.onnx:
                with open(args.onnx, 'wb') as f:
                    f.write(onnx_model.SerializeToString())
                print(f"XGBoost ONNX导出至: {args.onnx}")
    else:
        if args.svm_kernel != 'linear':
            print("警告: 非linear核无法导出到C++ LinearSVMModel，将使用普通SVC仅作评估")
            model = SVC(kernel=args.svm_kernel, class_weight='balanced', probability=True)
            scaler = StandardScaler()
            X = scaler.fit_transform(data)
            scores = cross_val_score(model, X, labels, cv=5, scoring='f1')
            print(f"5折交叉验证F1: {scores.mean():.4f} (±{scores.std():.4f})")
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("测试集评估:")
            print(classification_report(y_test, y_pred, target_names=["其他", "攀爬"], zero_division=0))
            print("混淆矩阵:")
            print(confusion_matrix(y_test, y_pred))
            print(f"\n{args.svm_kernel} 核无法按LinearSVMModel格式导出，仅展示评估结果。")
            return

        model, scaler = train_svm(data, labels)

    export_svm_model(model, scaler, args.output)
    export_scaler(scaler, args.scaler_out)
    print(f"训练完成，CV score已输出")


if __name__ == '__main__':
    main()