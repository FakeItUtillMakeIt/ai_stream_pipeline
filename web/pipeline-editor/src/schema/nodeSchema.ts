// 节点类型清单：驱动左侧面板 + 右侧属性表单。
// 字段用点路径表示嵌套(params 里的层级)，复杂结构用 json 编辑器兜底。
// 这些字段是照着各节点 configure()/configureImpl() 实际读取的键整理的。

export type FieldType =
  | 'string' | 'path' | 'number' | 'int' | 'bool' | 'enum'
  | 'stringList' | 'nodeRefList' | 'json';

export interface FieldDef {
  type: FieldType;
  label?: string;
  required?: boolean;
  default?: unknown;
  values?: string[];      // enum
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
}

export interface NodeTypeDef {
  label: string;
  category: string;
  color: string;
  fields: Record<string, FieldDef>;
}

export const CATEGORIES = [
  'source', 'decode', 'preprocess', 'infer', 'fusion',
  'track', 'alert', 'evidence', 'draw', 'sink', 'gate',
] as const;

export const CATEGORY_LABEL: Record<string, string> = {
  source: '源', decode: '解码', preprocess: '预处理', infer: '推理',
  fusion: '融合', track: '跟踪', alert: '告警', evidence: '证据',
  draw: '绘制', sink: '输出', gate: 'VLM闸门',
};

export const SCHEMA: Record<string, NodeTypeDef> = {
  // ===== source =====
  rtsp_source: {
    label: 'RTSP源', category: 'source', color: '#2f6fb0',
    fields: {
      url: { type: 'string', label: 'RTSP地址', required: true, hint: 'rtsp://...' },
      skip_frames: { type: 'int', label: '跳帧', default: 1, min: 1 },
    },
  },
  file_source: {
    label: '文件源', category: 'source', color: '#2f6fb0',
    fields: {
      url: { type: 'string', label: '文件地址', required: true, hint: 'file:///path/x.mp4' },
      loop: { type: 'bool', label: '循环', default: true },
      realtime: { type: 'bool', label: '实时节奏', default: true },
      skip_frames: { type: 'int', label: '跳帧', default: 1, min: 1 },
    },
  },

  // ===== decode =====
  ffmpeg_decode: {
    label: 'FFmpeg解码', category: 'decode', color: '#5a8',
    fields: {
      hw_decoder: { type: 'bool', label: '硬解码', default: true },
      output_bgr: { type: 'bool', label: '输出BGR', default: true },
      hw_decoder_type: { type: 'enum', label: '硬解类型', values: ['', 'h264_cuvid', 'hevc_cuvid'] },
    },
  },

  // ===== preprocess =====
  resize_normalize: {
    label: '预处理(缩放/归一)', category: 'preprocess', color: '#7a6',
    fields: {
      output_width: { type: 'int', label: '输出宽', required: true, default: 640 },
      output_height: { type: 'int', label: '输出高', required: true, default: 640 },
      keep_aspect_ratio: { type: 'bool', label: '保持比例(letterbox)', default: true },
      mean: { type: 'json', label: '均值[3]', hint: '如 [0.485,0.456,0.406]' },
      std: { type: 'json', label: '方差[3]', hint: '如 [0.229,0.224,0.225]' },
    },
  },

  // ===== infer =====
  detection_infer: {
    label: '目标检测', category: 'infer', color: '#b5b',
    fields: {
      'detector_config.model_path': { type: 'path', label: '模型(.engine)', required: true },
      'detector_config.conf_threshold': { type: 'number', label: '置信阈值', default: 0.25, min: 0, max: 1, step: 0.05 },
      'detector_config.nms_threshold': { type: 'number', label: 'NMS阈值', default: 0.4, min: 0, max: 1, step: 0.05 },
      'detector_config.input_size': { type: 'json', label: '输入尺寸', default: { width: 640, height: 640 } },
      'detector_config.batch_size': { type: 'int', label: 'batch', default: 1, min: 1 },
      'detector_config.model_class_names': { type: 'stringList', label: '类别名(顺序=通道)', hint: '顺序须与模型导出一致' },
      'detector_config.cuda_graph': { type: 'bool', label: 'CUDA Graph', default: true },
    },
  },
  pose_infer: {
    label: '姿态估计', category: 'infer', color: '#b5b',
    fields: {
      'model_path': { type: 'path', label: '模型(.engine)', required: true },
      'max_persons': { type: 'int', label: '最大人数', default: 8 },
      'conf_threshold': { type: 'number', label: '置信阈值', default: 0.3 },
    },
  },
  action_recognition_videomae: {
    label: '动作识别(VideoMAE)', category: 'infer', color: '#b5b',
    fields: {
      model_path: { type: 'path', label: '模型(.engine)', required: true },
      input_width: { type: 'int', label: '输入宽', default: 224 },
      input_height: { type: 'int', label: '输入高', default: 224 },
      num_frames: { type: 'int', label: 'clip帧数', default: 16 },
      frame_interval: { type: 'int', label: '采样间隔', default: 2 },
      window_size: { type: 'int', label: '滑窗大小', default: 16 },
      stride: { type: 'int', label: '滑窗步长', default: 8 },
      confidence_threshold: { type: 'number', label: '置信阈值', default: 0.7, min: 0, max: 1, step: 0.05 },
      batch_size: { type: 'int', label: 'batch', default: 1 },
      action_labels: { type: 'stringList', label: '动作标签', hint: '顺序须与模型输出类别一致' },
      passthrough_input: { type: 'bool', label: '透传原始帧', default: true, hint: '作并行旁支时设false' },
    },
  },
  relation_recognition: {
    label: '视觉关系', category: 'infer', color: '#b5b',
    fields: {
      model_path: { type: 'path', label: '关系模型(.engine)', required: true },
      predicate_bank: { type: 'path', label: '谓词表(.json)', required: true },
      img_size: { type: 'int', label: '输入边长', default: 448 },
      max_boxes: { type: 'int', label: '最大框数', default: 32 },
      predicates: { type: 'stringList', label: '激活谓词', default: ['beside', 'in front of', 'behind', 'on'] },
      score_threshold: { type: 'number', label: '关系分阈值', default: 0.2, min: 0, max: 1, step: 0.05 },
      box_confidence: { type: 'number', label: '框置信下限', default: 0.25, min: 0, max: 1, step: 0.05 },
      max_relations: { type: 'int', label: '单帧最大关系数', default: 256 },
      boundary_classes: { type: 'stringList', label: '边界类别', default: ['wall', 'gate'] },
      backend: { type: 'enum', label: '后端', values: ['auto', 'tensorrt', 'onnxruntime', 'cpu'], default: 'auto' },
      run_every_n_frames: { type: 'int', label: '隔帧运行', default: 1, min: 1 },
    },
  },

  // ===== fusion =====
  fusion: {
    label: '融合', category: 'fusion', color: '#c90',
    fields: {
      mode: { type: 'enum', label: '模式', values: ['action', 'detection_merge', 'object_level'], default: 'action' },
      detection_sources: { type: 'nodeRefList', label: '检测来源节点', hint: 'detection_merge 模式' },
      class_offsets: { type: 'json', label: 'class_id偏移', hint: '{"infer_yolov8":10}' },
      wait_timeout_ms: { type: 'int', label: '配对超时(ms)', default: 200 },
      cross_nms: { type: 'bool', label: '跨源NMS', default: false },
      nms_iou_threshold: { type: 'number', label: 'NMS IoU', default: 0.5, min: 0, max: 1, step: 0.05 },
      action_source: { type: 'string', label: '动作来源节点' },
      timestamp_threshold_ms: { type: 'int', label: '动作时间戳阈值(ms)', default: 500 },
      drop_non_inference: { type: 'bool', label: '丢弃非推理包', default: false },
      action_persist: { type: 'bool', label: '持续附加动作', default: false },
    },
  },

  // ===== track =====
  tracker: {
    label: '跟踪', category: 'track', color: '#88c',
    fields: {
      tracker_type: { type: 'enum', label: '算法', values: ['bytetrack', 'ocsort'], default: 'bytetrack' },
      sub_stream_id: { type: 'string', label: '关联源节点id' },
    },
  },

  // ===== alert =====
  alert: {
    label: '告警规则', category: 'alert', color: '#d55',
    fields: {
      process_type: { type: 'enum', label: '执行方式', values: ['parallel', 'sequence'], default: 'parallel' },
      rules: { type: 'json', label: '规则数组', required: true, hint: '[{"type":"...","params":{...}}]' },
    },
  },

  // ===== evidence =====
  evidence: {
    label: '证据(录像/截图)', category: 'evidence', color: '#0a8',
    fields: {
      video: { type: 'json', label: 'video', default: { enabled: true, pre_frames: 20, post_frames: 40, fps: 25, encoder: 'libx264', output_dir: './evidence/videos' } },
      snapshot: { type: 'json', label: 'snapshot', default: { enabled: true, output_dir: './evidence/snapshots' } },
      rollover: { type: 'json', label: 'rollover', default: { enabled: false, retention_hours: 72 } },
      ftp: { type: 'json', label: 'ftp', default: { enabled: false } },
    },
  },

  // ===== draw =====
  osd_draw: {
    label: '画面绘制(OSD)', category: 'draw', color: '#69c',
    fields: {
      snapshot: { type: 'json', label: 'snapshot', default: { enabled: false, interval: 100, dir: './draw_snapshot/' } },
      panel: { type: 'json', label: 'panel', default: { enabled: true } },
    },
  },

  // ===== sink =====
  rtmp_sink: {
    label: 'RTMP推流', category: 'sink', color: '#e80',
    fields: {
      output_url: { type: 'string', label: '推流地址', required: true, hint: 'rtmp://...' },
      encoder: { type: 'enum', label: '编码器', values: ['libx264', 'h264_nvenc', 'auto'] },
      bitrate: { type: 'int', label: '码率(kbps)', default: 4000 },
      output_width: { type: 'int', label: '输出宽', default: 1280 },
      output_height: { type: 'int', label: '输出高', default: 720 },
      queue: { type: 'json', label: 'queue', default: { capacity: 10, drop_policy: 'drop_oldest' } },
    },
  },
  mp4_save: {
    label: '存MP4', category: 'sink', color: '#e80',
    fields: {
      output_url: { type: 'string', label: '输出文件', required: true, hint: './output/x.mp4' },
      output_width: { type: 'int', label: '输出宽', default: 1280, hint: '须与来流一致，否则编码崩溃' },
      output_height: { type: 'int', label: '输出高', default: 720 },
      encoder: { type: 'enum', label: '编码器', values: ['libx264', 'h264_nvenc', 'auto'] },
      bitrate: { type: 'int', label: '码率(kbps)', default: 4000 },
    },
  },

  // ===== gate =====
  vlm_gate: {
    label: 'VLM告警闸门', category: 'gate', color: '#939',
    fields: {
      enabled: { type: 'bool', label: '启用', default: true },
      url: { type: 'string', label: 'OpenAI兼容基址', required: true, hint: 'http://ip:port/v1' },
      api_key: { type: 'string', label: 'API Key', hint: '支持 ${ENV}' },
      model: { type: 'string', label: '模型名', required: true },
      enable_thinking: { type: 'bool', label: '思考模式', default: false },
      max_tokens: { type: 'int', label: 'max_tokens', default: 512 },
      timeout_ms: { type: 'int', label: '超时(ms)', default: 3000 },
      retries: { type: 'int', label: '重试', default: 1 },
      max_side: { type: 'int', label: '送图最长边', default: 1024 },
      confidence_threshold: { type: 'number', label: '置信阈值τ', default: 0.6, min: 0, max: 1, step: 0.05 },
      cooldown_s: { type: 'int', label: '冷却(s)', default: 30 },
      unmatched_policy: { type: 'enum', label: '未配模板策略', values: ['auto', 'passthrough', 'hold'], default: 'auto' },
      review_dir: { type: 'string', label: '审计目录', default: './vlm_review' },
      prompts: { type: 'json', label: '各告警提示词', hint: '{"child_near_wall":{"question":"..."}}' },
    },
  },
  report: {
    label: '告警上报', category: 'gate', color: '#939',
    fields: {
      enabled: { type: 'bool', label: '启用', default: true },
      url: { type: 'string', label: 'Webhook地址', required: true },
      api_key: { type: 'string', label: 'API Key', hint: '支持 ${ENV}' },
      attach_image: { type: 'bool', label: '附带base64图', default: true },
      image_field: { type: 'string', label: '图片字段名', default: 'snapshot_base64' },
      image_max_side: { type: 'int', label: '图片下采样(0=原样)', default: 0 },
      timeout_ms: { type: 'int', label: '超时(ms)', default: 3000 },
      source: { type: 'string', label: '来源标识', default: 'ai_stream' },
    },
  },
};

export function nodeTypesInCategory(cat: string): string[] {
  return Object.keys(SCHEMA).filter((t) => SCHEMA[t].category === cat);
}

// 依据 schema 生成一个节点的默认 params（只含带 default 的字段）
export function defaultParams(type: string): Record<string, unknown> {
  const def = SCHEMA[type];
  const params: Record<string, unknown> = {};
  if (!def) return params;
  for (const [path, f] of Object.entries(def.fields)) {
    if (f.default !== undefined) {
      // 深拷贝默认值，避免多节点共享
      setDefault(params, path, JSON.parse(JSON.stringify(f.default)));
    }
  }
  return params;
}

function setDefault(obj: Record<string, unknown>, path: string, val: unknown) {
  const keys = path.split('.');
  let cur: any = obj;
  for (let i = 0; i < keys.length - 1; i++) {
    if (typeof cur[keys[i]] !== 'object' || cur[keys[i]] === null) cur[keys[i]] = {};
    cur = cur[keys[i]];
  }
  cur[keys[keys.length - 1]] = val;
}
