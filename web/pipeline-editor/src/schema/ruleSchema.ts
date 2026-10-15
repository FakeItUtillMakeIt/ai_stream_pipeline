// 告警规则子 schema：每个规则类型的参数（对齐各 *_rule.cpp 的 initialize() 实际读取的键）。
// 通用：name、rule_zones（多边形数组）。其余按类型。
import { FieldDef } from './nodeSchema';

export interface RuleTypeDef {
  label: string;
  fields: Record<string, FieldDef>;
}

// 所有规则都有的公共字段
const COMMON: Record<string, FieldDef> = {
  name: { type: 'string', label: '规则名', required: true },
  rule_zones: { type: 'json', label: '区域(多边形数组)', hint: '[[[x,y],[x,y],...]]；空=全域' },
};

function withCommon(extra: Record<string, FieldDef>): Record<string, FieldDef> {
  return { ...COMMON, ...extra };
}

export const RULE_SCHEMA: Record<string, RuleTypeDef> = {
  person_intrusion: { label: '人员入侵', fields: withCommon({}) },
  person_loitering: {
    label: '人员徘徊', fields: withCommon({
      target_class: { type: 'string', label: '目标类别', default: 'person' },
      loiter_duration_ms: { type: 'int', label: '滞留时长(ms)', default: 10000 },
      leave_grace_ms: { type: 'int', label: '离开宽限(ms)', default: 2000 },
      duration_ms: { type: 'int', label: '确认窗口(ms)', default: 1000 },
    }),
  },
  child_near_boundary: {
    label: '儿童靠近边界', fields: withCommon({
      child_class: { type: 'string', label: '儿童类别', default: 'child' },
      boundary_classes: { type: 'stringList', label: '边界类别', default: ['wall', 'gate'] },
      relation_threshold: { type: 'number', label: '关系分阈值', default: 0.2, min: 0, max: 1, step: 0.05 },
      geometric_fallback: { type: 'bool', label: '几何回退', default: false },
      max_pixel_gap: { type: 'int', label: '最大像素间距', default: 60 },
      duration_ms: { type: 'int', label: '持续时长(ms)', default: 1000 },
      max_disappear_count: { type: 'int', label: '消失容忍帧', default: 5 },
    }),
  },
  fighting: {
    label: '打架', fields: withCommon({
      action_recongnition_mode: { type: 'enum', label: '识别方式', values: ['pose', 'model'], default: 'pose' },
      fight_hold_ms: { type: 'int', label: '一次保持(ms)', default: 0 },
    }),
  },
  climbing: {
    label: '攀爬', fields: withCommon({
      action_recongnition_mode: { type: 'enum', label: '识别方式', values: ['pose', 'model'], default: 'pose' },
      climb_hold_ms: { type: 'int', label: '一次保持(ms)', default: 0 },
    }),
  },
  falling: {
    label: '跌倒(过程)', fields: withCommon({
      person_class: { type: 'string', label: '人员类别', default: 'person' },
      down_class: { type: 'string', label: '倒地类别' },
      down_confirm_ms: { type: 'int', label: '倒地确认(ms)' },
      track_timeout_ms: { type: 'int', label: '跟踪超时(ms)' },
      alert_duration_ms: { type: 'int', label: '告警时长(ms)' },
    }),
  },
  fall_down: { label: '跌倒(检测框)', fields: withCommon({}) },
  human_gathering: {
    label: '人员聚集', fields: withCommon({
      gathering_thresh: { type: 'int', label: '聚集人数阈值', default: 3 },
    }),
  },
  fire_lane_occupancy: {
    label: '消防通道占用', fields: withCommon({
      reference_image: { type: 'path', label: '参考图' },
      similarity_threshold: { type: 'number', label: '相似度阈值', min: 0, max: 1, step: 0.05 },
      confirm_frames: { type: 'int', label: '确认帧数' },
      init_frames: { type: 'int', label: '初始帧数' },
      update_frames: { type: 'int', label: '更新帧数' },
    }),
  },
  absence: { label: '离岗', fields: withCommon({}) },
  sleeping_on_duty: { label: '睡岗', fields: withCommon({}) },
  sleep_on_duty: { label: '睡岗', fields: withCommon({}) },
  smoking: { label: '吸烟', fields: withCommon({}) },
  phone_call: { label: '打电话', fields: withCommon({}) },
  moving_phone_call: { label: '移动中打电话', fields: withCommon({}) },
  missing_helmet: { label: '未戴安全帽', fields: withCommon({}) },
  missing_work_clothes: { label: '未穿工作服', fields: withCommon({}) },
  missing_safety_belt: { label: '未系安全带', fields: withCommon({}) },
  rising: { label: ' rising', fields: withCommon({}) },
  photographer: { label: '揽拍', fields: withCommon({}) },
  unlicensed_verdor: { label: '无证摊贩', fields: withCommon({}) },
  discover_crystal: { label: '发现结晶', fields: withCommon({}) },
  discover_visible_fire: { label: '发现火焰', fields: withCommon({}) },
  discover_smoke: { label: '发现烟雾', fields: withCommon({}) },
  discover_hose_cutoff: { label: '发现软管断流', fields: withCommon({}) },
};

export function ruleTypes(): string[] { return Object.keys(RULE_SCHEMA); }

// 生成一条新规则的默认 params
export function defaultRule(type: string): { type: string; params: Record<string, unknown> } {
  const def = RULE_SCHEMA[type];
  const params: Record<string, unknown> = { name: type };
  if (def) {
    for (const [k, f] of Object.entries(def.fields)) {
      if (f.default !== undefined && k !== 'name') params[k] = JSON.parse(JSON.stringify(f.default));
    }
  }
  return { type, params };
}
