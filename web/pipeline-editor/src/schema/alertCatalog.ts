// 告警场景目录：与后端 src/core/alert_catalog.cpp 保持一致。
// 分类用于页面分组；requiredModels/Classes/Labels 用于禁用未绑定模型的规则。
import { NodeDef } from '../types';

export const ITEM_PERSON_BEHAVIOR = 'ITEM_PERSON_BEHAVIOR';
export const ITEM_SCENE_RECOGNITION = 'ITEM_SCENE_RECOGNITION';
export const ITEM_RELATION_RECOGNITION = 'ITEM_RELATION_RECOGNITION';
export const ITEM_ACTION_RECOGNITION = 'ITEM_ACTION_RECOGNITION';

export interface AlertCategoryDef {
  id: string;
  label: string;
  order: number;
}

export const ALERT_CATEGORIES: AlertCategoryDef[] = [
  { id: ITEM_PERSON_BEHAVIOR, label: '人员行为', order: 1 },
  { id: ITEM_SCENE_RECOGNITION, label: '场景识别', order: 2 },
  { id: ITEM_RELATION_RECOGNITION, label: '关系识别', order: 3 },
  { id: ITEM_ACTION_RECOGNITION, label: '动作识别', order: 4 },
];

export interface AlertRuleMeta {
  category: string;
  severity: 'P0' | 'P1' | 'P2';
  requiredModels: string[];
  requiredClasses: string[];
  requiredLabels: string[];
}

export const RULE_META: Record<string, AlertRuleMeta> = {
  // 人员行为 / 劳保（9cls）
  person_intrusion: { category: ITEM_PERSON_BEHAVIOR, severity: 'P0', requiredModels: ['person_behavior_det'], requiredClasses: ['person'], requiredLabels: [] },
  person_loitering: { category: ITEM_PERSON_BEHAVIOR, severity: 'P2', requiredModels: ['person_behavior_det'], requiredClasses: ['person'], requiredLabels: [] },
  absence: { category: ITEM_PERSON_BEHAVIOR, severity: 'P1', requiredModels: ['person_behavior_det'], requiredClasses: ['person'], requiredLabels: [] },
  sleep_on_duty: { category: ITEM_PERSON_BEHAVIOR, severity: 'P1', requiredModels: ['person_behavior_det'], requiredClasses: ['person', 'sleeping'], requiredLabels: [] },
  smoking: { category: ITEM_PERSON_BEHAVIOR, severity: 'P2', requiredModels: ['person_behavior_det'], requiredClasses: ['person', 'smoking'], requiredLabels: [] },
  phone_call: { category: ITEM_PERSON_BEHAVIOR, severity: 'P2', requiredModels: ['person_behavior_det'], requiredClasses: ['person', 'phone'], requiredLabels: [] },
  moving_phone_call: { category: ITEM_PERSON_BEHAVIOR, severity: 'P2', requiredModels: ['person_behavior_det'], requiredClasses: ['person', 'phone'], requiredLabels: [] },
  human_gathering: { category: ITEM_PERSON_BEHAVIOR, severity: 'P2', requiredModels: ['person_behavior_det'], requiredClasses: ['person'], requiredLabels: [] },
  falling: { category: ITEM_PERSON_BEHAVIOR, severity: 'P0', requiredModels: ['person_behavior_det'], requiredClasses: ['person', 'fall_down'], requiredLabels: [] },
  fall_down: { category: ITEM_PERSON_BEHAVIOR, severity: 'P0', requiredModels: ['person_behavior_det'], requiredClasses: ['person', 'fall_down'], requiredLabels: [] },
  rising: { category: ITEM_PERSON_BEHAVIOR, severity: 'P2', requiredModels: ['person_behavior_det'], requiredClasses: ['person'], requiredLabels: [] },
  photographer: { category: ITEM_PERSON_BEHAVIOR, severity: 'P2', requiredModels: ['person_behavior_det'], requiredClasses: ['person'], requiredLabels: [] },
  unlicensed_verdor: { category: ITEM_PERSON_BEHAVIOR, severity: 'P2', requiredModels: ['person_behavior_det'], requiredClasses: ['person'], requiredLabels: [] },
  missing_helmet: { category: ITEM_PERSON_BEHAVIOR, severity: 'P1', requiredModels: ['person_behavior_det'], requiredClasses: ['person', 'head', 'helmet'], requiredLabels: [] },
  missing_work_clothes: { category: ITEM_PERSON_BEHAVIOR, severity: 'P1', requiredModels: ['person_behavior_det'], requiredClasses: ['person', 'work_clothes'], requiredLabels: [] },
  missing_safety_belt: { category: ITEM_PERSON_BEHAVIOR, severity: 'P1', requiredModels: ['person_behavior_det'], requiredClasses: ['person', 'safety_belt'], requiredLabels: [] },

  // 动作识别（9cls + action）
  climbing: { category: ITEM_ACTION_RECOGNITION, severity: 'P1', requiredModels: ['person_behavior_det', 'action'], requiredClasses: ['person'], requiredLabels: ['climbing'] },
  fighting: { category: ITEM_ACTION_RECOGNITION, severity: 'P0', requiredModels: ['person_behavior_det', 'action'], requiredClasses: ['person'], requiredLabels: ['fighting'] },

  // 场景识别（scene.engine）
  discover_crystal: { category: ITEM_SCENE_RECOGNITION, severity: 'P1', requiredModels: ['scene_det'], requiredClasses: ['crystal'], requiredLabels: [] },
  discover_visible_fire: { category: ITEM_SCENE_RECOGNITION, severity: 'P0', requiredModels: ['scene_det'], requiredClasses: ['fire'], requiredLabels: [] },
  discover_smoke: { category: ITEM_SCENE_RECOGNITION, severity: 'P0', requiredModels: ['scene_det'], requiredClasses: ['smoke'], requiredLabels: [] },
  discover_hose_cutoff: { category: ITEM_SCENE_RECOGNITION, severity: 'P1', requiredModels: ['scene_det'], requiredClasses: ['hose'], requiredLabels: [] },
  fire_lane_occupancy: { category: ITEM_SCENE_RECOGNITION, severity: 'P2', requiredModels: [], requiredClasses: [], requiredLabels: [] },

  // 关系识别（relation + boundary）
  child_near_boundary: { category: ITEM_RELATION_RECOGNITION, severity: 'P0', requiredModels: ['relation', 'boundary_det'], requiredClasses: ['child', 'wall', 'gate'], requiredLabels: [] },
};

export function categoryLabel(id: string): string {
  return ALERT_CATEGORIES.find((c) => c.id === id)?.label || id;
}

export interface ModelCapabilities {
  roles: Set<string>;
  categories: Set<string>;
  byRole: Record<string, { classes: Set<string>; labels: Set<string> }>;
}

function asArray(v: unknown): string[] {
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === 'string');
  if (typeof v === 'string' && v) return [v];
  return [];
}

function normalizeCategory(v: string): string {
  if (!v) return v;
  if (v.startsWith('ITEM_')) return v;
  if (v === 'person' || v === 'person_behavior') return ITEM_PERSON_BEHAVIOR;
  if (v === 'scene' || v === 'scene_recognition') return ITEM_SCENE_RECOGNITION;
  if (v === 'relation' || v === 'relation_recognition') return ITEM_RELATION_RECOGNITION;
  if (v === 'action' || v === 'action_recognition') return ITEM_ACTION_RECOGNITION;
  return v;
}

export function collectModelCapabilities(nodes: NodeDef[]): ModelCapabilities {
  const caps: ModelCapabilities = { roles: new Set(), categories: new Set(), byRole: {} };
  const add = (role: string, categories: string[], classes: string[], labels: string[]) => {
    if (!role && categories.length === 0) return;
    if (role) caps.roles.add(role);
    for (const c of categories) caps.categories.add(normalizeCategory(c));
    if (!role) return;
    const slot = caps.byRole[role] || (caps.byRole[role] = { classes: new Set(), labels: new Set() });
    for (const c of classes) slot.classes.add(c);
    for (const l of labels) slot.labels.add(l);
  };

  for (const node of nodes) {
    const type = node.type;
    if (!['detection_infer', 'pose_infer', 'relation_recognition', 'action_recognition_videomae'].includes(type)) continue;
    const p: any = node.params || {};
    const dc: any = p.detector_config && typeof p.detector_config === 'object' ? p.detector_config : {};
    let role = String(dc.model_role ?? p.model_role ?? '');
    let categories = [...asArray(dc.model_categories), ...asArray(p.model_categories)].map(normalizeCategory);
    let classes = [...asArray(dc.model_class_names), ...asArray(p.model_class_names)];
    let labels = [...asArray(p.action_labels), ...asArray(dc.action_labels)];

    if (type === 'relation_recognition') {
      if (!role) role = 'relation';
      classes = [...classes, ...asArray(p.boundary_classes)];
      if (!categories.length) categories = [ITEM_RELATION_RECOGNITION];
    } else if (type === 'action_recognition_videomae') {
      if (!role) role = 'action';
      if (!categories.length) categories = [ITEM_ACTION_RECOGNITION];
    } else if (type === 'pose_infer') {
      if (!role) role = 'pose';
      if (!categories.length) categories = [ITEM_PERSON_BEHAVIOR];
    } else {
      if (!categories.length) {
        if (role === 'person_behavior_det') categories = [ITEM_PERSON_BEHAVIOR];
        else if (role === 'scene_det') categories = [ITEM_SCENE_RECOGNITION];
        else if (role === 'boundary_det' || role === 'relation') categories = [ITEM_RELATION_RECOGNITION];
        else if (role === 'action') categories = [ITEM_ACTION_RECOGNITION];
        else if (classes.some((c) => ['fire', 'smoke', 'crystal', 'hose'].includes(c))) categories = [ITEM_SCENE_RECOGNITION];
        else if (classes.some((c) => ['person', 'head', 'helmet', 'work_clothes', 'safety_belt', 'smoking', 'phone', 'sleeping', 'fall_down'].includes(c))) categories = [ITEM_PERSON_BEHAVIOR];
        else if (classes.some((c) => ['child', 'wall', 'gate'].includes(c))) categories = [ITEM_RELATION_RECOGNITION];
      }
      if (!role) {
        if (categories.includes(ITEM_SCENE_RECOGNITION)) role = 'scene_det';
        else if (categories.includes(ITEM_PERSON_BEHAVIOR)) role = 'person_behavior_det';
        else if (categories.includes(ITEM_RELATION_RECOGNITION)) role = 'boundary_det';
        else if (categories.includes(ITEM_ACTION_RECOGNITION)) role = 'action';
      }
    }

    add(role, categories, classes, labels);
  }
  return caps;
}

export function ruleUnavailableReason(type: string, caps: ModelCapabilities): string | null {
  const meta = RULE_META[type];
  if (!meta) return null;
  if (meta.requiredModels.length === 0) return null;

  if (!caps.categories.has(meta.category)) {
    return `当前模板缺少「${categoryLabel(meta.category)}」模型`;
  }
  for (const role of meta.requiredModels) {
    if (!caps.roles.has(role)) return `缺少模型：${role}`;
  }
  for (const cls of meta.requiredClasses) {
    const ok = meta.requiredModels.some((role) => caps.byRole[role]?.classes.has(cls));
    if (!ok) return `缺少模型类别：${cls}`;
  }
  for (const label of meta.requiredLabels) {
    const ok = meta.requiredModels.some((role) => caps.byRole[role]?.labels.has(label));
    if (!ok) return `动作模型缺少标签：${label}`;
  }
  return null;
}

export function severityTone(severity: string): 'p0' | 'p1' | 'p2' {
  if (severity === 'P0') return 'p0';
  if (severity === 'P1') return 'p1';
  return 'p2';
}
