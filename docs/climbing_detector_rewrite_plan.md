# ClimbingDetector 重写实现计划（v2）

## 目标
基于确认的骨架规则重写 climbing_detector.h/.cpp，区分攀爬与跳舞/打架。

## 文件改动
- `src/rules/alert/detector/climbing_detector.h` — 新增函数声明 + Config参数
- `src/rules/alert/detector/climbing_detector.cpp` — 重写所有检测函数

---

## 一、Config 新增参数

```cpp
// === 静态骨架 ===
// 手高于肩
float arm_raise_offset = 2.0f;           // 手腕需高于肩膀的像素偏移
// 手臂弯曲
float arm_bend_min = 60.0f;              // 手臂弯曲角度下限（度）
float arm_bend_max = 130.0f;             // 手臂弯曲角度上限（度）
// 膝盖抬起
float knee_bend_min = 60.0f;             // 腿部弯曲角度下限（度）
float knee_bend_max = 130.0f;            // 腿部弯曲角度上限（度）
// 重心抬高/蜷缩
float center_raise_threshold = 15.0f;    // 重心抬高像素阈值（相对历史均值）
float stretch_compress_ratio = 0.7f;     // 蜷缩比值阈值（头-踝/bbox_h）

// === 身体朝向 ===
// 身体倾斜
float tilt_min = 20.0f;                  // 躯干倾斜最小角度（度）
float tilt_max = 70.0f;                  // 躯干倾斜最大角度（度）
// 四肢张开
float limb_span_threshold = 0.6f;        // 四肢张开归一化阈值

// === 动态特征 ===
// 交替抬手抬脚
float alternation_ratio_threshold = 0.4f; // 交替帧比例阈值
int alternation_window = 10;              // 交替检测窗口帧数
// 整体向上移动
float ascent_slope_threshold = -2.0f;    // 上升斜率阈值（像素/帧）
int ascent_min_frames = 5;               // 最少连续上升帧数
```

---

## 二、辅助函数（private，新增）

### calculateAngle(a, b, c) → float
- 计算三点夹角（b为顶点）
- `angle = atan2(a.y-b.y, a.x-b.x) - atan2(c.y-b.y, c.x-b.x)`
- 归一化到 0°~180°

### calculateOscillation(y_history) → float
- 统计方向变化次数 / 总帧数
- 返回 0.0~1.0

### calculateLateralMovement(center_history) → float
- X轴总位移 / Y轴总位移
- 返回 0.0~1.0

### calculateMovementBurst(center_history) → float
- 相邻帧速度方差
- 返回 0.0~1.0

---

## 三、重写检测函数

### A. 静态特征（单帧）

#### detectHandAboveShoulder(det, out_conf) → bool
```
至少一只手的手腕Y < 同侧肩膀Y - arm_raise_offset
置信度 = 1.0（单手）或 1.0（双手，额外加分）
```

#### detectArmBend(det, out_conf) → bool
```
对左右手臂分别计算：calculateAngle(shoulder, elbow, wrist)
至少一只手臂角度在 [arm_bend_min, arm_bend_max] 范围内
置信度 = min(1.0, 弯曲偏离90°的程度)
```

#### detectKneeRaise(det, out_conf) → bool
```
对左右腿分别计算：calculateAngle(hip, knee, ankle)
至少一条腿角度在 [knee_bend_min, knee_bend_max] 范围内
置信度 = min(1.0, 弯曲偏离90°的程度)
```

#### detectCenterRaise(det, state, out_conf) → bool
```
center_y = 所有可见关键点的平均Y
stretch_ratio = (ankle_y - head_y) / bbox_h （取最大ankle和最小head）
条件：
  center_y < state.avg_center_y - center_raise_threshold （重心抬高）
  或 stretch_ratio < stretch_compress_ratio （蜷缩）
置信度 = 根据偏离程度计算
更新 state.avg_center_y（滑动平均）
```

#### detectBodyTilt(det, out_conf) → bool
```
shoulder_mid = ((ls.x+rs.x)/2, (ls.y+rs.y)/2)
hip_mid = ((lh.x+rh.x)/2, (lh.y+rh.y)/2)
angle = atan2(shoulder_mid.x - hip_mid.x, hip_mid.y - shoulder_mid.y) * 180/π
条件：abs(angle) 在 [tilt_min, tilt_max] 范围内
置信度 = min(1.0, 偏离垂直的程度)
```

#### detectLimbSpan(det, out_conf) → bool
```
limb_span = dist(lw, rw) + dist(la, ra)
bbox_diag = sqrt(w² + h²)
normalize = limb_span / bbox_diag
条件：normalize > limb_span_threshold
置信度 = min(1.0, normalize / (limb_span_threshold * 2))
```

### B. 动态特征（多帧）

#### detectAlternatingLimb(det, state, out_conf) → bool
```
对近N帧（alternation_window），统计：
  left_dy = left_wrist_y[t] - left_wrist_y[t-1]
  right_dy = right_wrist_y[t] - right_wrist_y[t-1]
  如果 left_dy * right_dy < 0 → 交替帧+1
alternation_ratio = 交替帧数 / 总帧数
条件：alternation_ratio > alternation_ratio_threshold
置信度 = alternation_ratio
```

#### detectOverallAscent(det, state, out_conf) → bool
```
对近N帧的center_y做线性回归
斜率 < ascent_slope_threshold → 持续上升
要求连续ascent_min_frames帧满足
置信度 = min(1.0, abs(斜率) / abs(ascent_slope_threshold))
```

---

## 四、过滤器函数（惩罚得分）

### filterByOscillation(state) → float
- calculateOscillation(center_y_history)
- > 0.5 → 0.3; > 0.3 → 0.7; else → 1.0

### filterByLateralMovement(state) → float
- calculateLateralMovement(center_history)
- > 0.5 → 0.3; > 0.3 → 0.7; else → 1.0

### filterByMovementBurst(state) → float
- calculateMovementBurst(center_history)
- > 0.6 → 0.3; > 0.4 → 0.7; else → 1.0

---

## 五、analyzeFrame 评分逻辑

```
static_score = hand_above_shoulder*0.2 + arm_bend*0.2 + knee_raise*0.2
             + center_raise*0.15 + body_tilt*0.15 + limb_span*0.1

dynamic_score = alternating*0.5 + ascent*0.5

raw_score = static_score*0.4 + dynamic_score*0.6

penalty = filter_oscillation * filter_lateral * filter_burst

final_score = raw_score * penalty

硬性约束：ascent 必须为 true，否则 final_score = 0
is_suspicious = final_score > climb_score_threshold
```

---

## 六、状态机
不改动，保持现有 IDLE→SUSPICIOUS→CLIMBING→COOLDOWN 逻辑。

## 七、验证
编译通过后测试：
- 攀爬场景 → CLIMBING 状态
- 跳舞场景 → 被振荡/横移过滤器惩罚，不触发
- 打架场景 → 被突变/横移过滤器惩罚，不触发
