from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import math
import argparse
import sys
from boxmot.trackers.ocsort.ocsort import OcSort


# ========== 参数配置 ==========
class Config:
    # 检测参数
    DET_CONF = 0.5
    POSE_CONF = 0.5

    # 跟踪参数
    TRACK_CONFIG = {
        'det_thresh': 0.5,
        'max_age': 30,
        'min_hits': 3,
        'iou_threshold': 0.3,
        'delta_t': 3,
        'asso_func': 'iou',
        'inertia': 0.2,
        'use_byte': False,
    }

    # 打架检测参数
    PUNCH_SPEED_THRESHOLD = 25
    FALL_ASPECT_RATIO_THRESHOLD = 1.2
    FIGHT_FRAMES_THRESHOLD = 5
    HISTORY_FRAMES = 10


# COCO数据集关键点连接关系
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

COLORS = {'head': (0, 255, 0), 'arms': (255, 0, 0), 'body': (0, 165, 255), 'legs': (0, 0, 255)}
KEYPOINT_COLORS = [(0, 255, 0)] * 5 + [(255, 0, 0)] * 6 + [(0, 165, 255)] * 2 + [(0, 0, 255)] * 4


# ========== 挥拳检测器 ==========
class PunchDetector:
    def __init__(self, history_frames=10, speed_threshold=25):
        self.wrist_history = {}
        self.history_frames = history_frames
        self.speed_threshold = speed_threshold
        self.punch_frames = {}

    def detect(self, keypoints, track_id):
        if keypoints is None or len(keypoints) < 17:
            return False, 0.0

        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        is_punch = False
        punch_confidence = 0.0

        if left_wrist[0] > 0 and left_wrist[1] > 0:
            is_punch_left, conf_left = self._check_arm_movement(left_wrist, track_id, 'left')
            if is_punch_left:
                is_punch = True
                punch_confidence = max(punch_confidence, conf_left)

        if right_wrist[0] > 0 and right_wrist[1] > 0:
            is_punch_right, conf_right = self._check_arm_movement(right_wrist, track_id, 'right')
            if is_punch_right:
                is_punch = True
                punch_confidence = max(punch_confidence, conf_right)

        if track_id not in self.punch_frames:
            self.punch_frames[track_id] = 0

        if is_punch:
            self.punch_frames[track_id] += 1
        else:
            self.punch_frames[track_id] = max(0, self.punch_frames[track_id] - 1)

        return is_punch, punch_confidence

    def _check_arm_movement(self, wrist, track_id, arm_type):
        if track_id not in self.wrist_history:
            self.wrist_history[track_id] = {
                'left': deque(maxlen=self.history_frames),
                'right': deque(maxlen=self.history_frames)
            }

        history = self.wrist_history[track_id][arm_type]
        current_pos = (wrist[0], wrist[1])
        history.append(current_pos)

        if len(history) < 3:
            return False, 0.0

        speeds = []
        for i in range(len(history)-1):
            prev = history[i]
            curr = history[i+1]
            displacement = math.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)
            speeds.append(displacement)

        if not speeds:
            return False, 0.0

        avg_speed = np.mean(speeds)

        if len(history) >= 3:
            v1 = (history[-2][0] - history[-3][0], history[-2][1] - history[-3][1])
            v2 = (history[-1][0] - history[-2][0], history[-1][1] - history[-2][1])
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
            norm2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if norm1 > 0 and norm2 > 0:
                cos_angle = dot / (norm1 * norm2)
                angle_change = abs(math.degrees(math.acos(max(-1, min(1, cos_angle)))))
            else:
                angle_change = 0
        else:
            angle_change = 0

        confidence = 0.0
        if avg_speed > self.speed_threshold:
            confidence = min(1.0, avg_speed / (self.speed_threshold * 2))
            if angle_change < 45:
                confidence = min(1.0, confidence * 1.2)

        return confidence > 0.5, confidence

    def cleanup(self, active_ids):
        """清理不再活跃的track_id，防止内存泄漏"""
        for tid in list(self.wrist_history.keys()):
            if tid not in active_ids:
                del self.wrist_history[tid]
        for tid in list(self.punch_frames.keys()):
            if tid not in active_ids:
                del self.punch_frames[tid]


# ========== 摔倒检测器 ==========
class FallDetector:
    def __init__(self, aspect_ratio_threshold=1.2):
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.fall_frames = {}

    def detect(self, bbox, track_id):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if height == 0:
            return False

        aspect_ratio = width / height

        if track_id not in self.fall_frames:
            self.fall_frames[track_id] = 0

        is_fall = aspect_ratio > self.aspect_ratio_threshold

        if is_fall:
            self.fall_frames[track_id] += 1
        else:
            self.fall_frames[track_id] = max(0, self.fall_frames[track_id] - 1)

        return is_fall

    def cleanup(self, active_ids):
        for tid in list(self.fall_frames.keys()):
            if tid not in active_ids:
                del self.fall_frames[tid]


# ========== 多人交互检测器 ==========
class InteractionDetector:
    def __init__(self, distance_threshold=80):
        self.distance_threshold = distance_threshold

    def detect(self, keypoints_list, track_ids):
        interactions = []

        for i in range(len(keypoints_list)):
            for j in range(i+1, len(keypoints_list)):
                kp1 = keypoints_list[i]
                kp2 = keypoints_list[j]
                tid1 = track_ids[i]
                tid2 = track_ids[j]

                if kp1 is None or kp2 is None:
                    continue

                left_wrist1 = kp1[9] if len(kp1) > 9 else None
                right_wrist1 = kp1[10] if len(kp1) > 10 else None
                left_wrist2 = kp2[9] if len(kp2) > 9 else None
                right_wrist2 = kp2[10] if len(kp2) > 10 else None

                min_distance = float('inf')
                contact_parts = []

                if left_wrist1 is not None and left_wrist2 is not None:
                    if left_wrist1[0] > 0 and left_wrist2[0] > 0:
                        dist = math.sqrt((left_wrist1[0]-left_wrist2[0])**2 + (left_wrist1[1]-left_wrist2[1])**2)
                        if dist < min_distance:
                            min_distance = dist
                        if dist < self.distance_threshold:
                            contact_parts.append('wrist')

                if right_wrist1 is not None and right_wrist2 is not None:
                    if right_wrist1[0] > 0 and right_wrist2[0] > 0:
                        dist = math.sqrt((right_wrist1[0]-right_wrist2[0])**2 + (right_wrist1[1]-right_wrist2[1])**2)
                        if dist < min_distance:
                            min_distance = dist
                        if dist < self.distance_threshold:
                            contact_parts.append('wrist')

                head1 = kp1[0] if len(kp1) > 0 else None
                head2 = kp2[0] if len(kp2) > 0 else None
                if head1 is not None and head2 is not None:
                    if head1[0] > 0 and head2[0] > 0:
                        dist = math.sqrt((head1[0]-head2[0])**2 + (head1[1]-head2[1])**2)
                        if dist < min_distance:
                            min_distance = dist
                        if dist < self.distance_threshold:
                            contact_parts.append('head')

                if min_distance < self.distance_threshold * 1.5:
                    interactions.append({
                        'person1': tid1,
                        'person2': tid2,
                        'distance': min_distance,
                        'contact_parts': contact_parts,
                        'is_contact': len(contact_parts) > 0
                    })

        return interactions


# ========== 跟踪管理器 ==========
class TrackManager:
    def __init__(self, config):
        self.tracker = OcSort(
            det_thresh=config['det_thresh'],
            max_age=config['max_age'],
            min_hits=config['min_hits'],
            iou_threshold=config['iou_threshold'],
            delta_t=config['delta_t'],
            asso_func=config['asso_func'],
            inertia=config['inertia'],
            use_byte=config['use_byte']
        )
        self.track_history = {}
        self.max_history = 30

    def update(self, detections, keypoints_list, frame):
        """
        detections: numpy array of shape (N, 6) [x1, y1, x2, y2, conf, cls]
        keypoints_list: list of (17, 2) arrays or None
        frame: 原始图像 (numpy array)
        """
        if len(detections) == 0:
            tracked_dets = self.tracker.update(np.empty((0, 6)), frame)
            return [], []

        # 更新跟踪器，需要传入图像 frame
        tracked_dets = self.tracker.update(detections, frame)

        tracked_results = []
        tracked_keypoints = []

        if len(tracked_dets) > 0:
            for track_det in tracked_dets:
                # boxmot 返回格式: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
                track_id = int(track_det[4])
                bbox = track_det[:4].astype(int)

                # 找到对应的关键点（通过IOU匹配）
                best_iou = 0
                best_kp = None

                for i, det in enumerate(detections):
                    iou = self._calculate_iou(bbox, det[:4])
                    if iou > best_iou and iou > 0.5:
                        best_iou = iou
                        best_kp = keypoints_list[i] if i < len(keypoints_list) else None

                if best_kp is not None:
                    tracked_results.append({
                        'track_id': track_id,
                        'bbox': bbox,
                        'confidence': track_det[5]
                    })
                    tracked_keypoints.append(best_kp)

                    if track_id not in self.track_history:
                        self.track_history[track_id] = {
                            'bboxes': deque(maxlen=self.max_history),
                            'keypoints': deque(maxlen=self.max_history)
                        }
                    self.track_history[track_id]['bboxes'].append(bbox)
                    self.track_history[track_id]['keypoints'].append(best_kp)

        return tracked_results, tracked_keypoints

    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0


# ========== 打架检测主类 ==========
class FightingDetector:
    def __init__(self, track_config=Config.TRACK_CONFIG):
        self.punch_detector = PunchDetector(
            history_frames=Config.HISTORY_FRAMES,
            speed_threshold=Config.PUNCH_SPEED_THRESHOLD
        )
        self.fall_detector = FallDetector(
            aspect_ratio_threshold=Config.FALL_ASPECT_RATIO_THRESHOLD
        )
        self.interaction_detector = InteractionDetector(distance_threshold=80)
        self.track_manager = TrackManager(track_config)

        self.fighting_tracks = {}
        self.fight_history = deque(maxlen=30)

    def process_frame(self, detections, keypoints_list, frame):
        # 修复：空检测时维度统一为 (0, 6)
        if len(detections) > 0:
            dets_array = np.array(detections)
            if dets_array.shape[1] != 6:
                # 如果传入的是5列，补一列类别
                if dets_array.shape[1] == 5:
                    cls_col = np.zeros((dets_array.shape[0], 1))
                    dets_array = np.hstack([dets_array, cls_col])
        else:
            dets_array = np.empty((0, 6))
        # 更新跟踪，传入原始图像 frame
        tracked_results, tracked_keypoints = self.track_manager.update(dets_array, keypoints_list, frame)
        # 获取活跃ID用于内存清理
        active_ids = {t['track_id'] for t in tracked_results}

        if len(tracked_results) == 0:
            # 清理不活跃ID
            self._cleanup_inactive(active_ids)
            return {'is_fighting': False, 'fight_score': 0, 'events': []}

        events = []
        fight_score = 0.0

        for i, track in enumerate(tracked_results):
            track_id = track['track_id']
            bbox = track['bbox']
            keypoints = tracked_keypoints[i] if i < len(tracked_keypoints) else None

            is_punch, punch_conf = self.punch_detector.detect(keypoints, track_id)
            if is_punch:
                events.append({
                    'type': 'punch',
                    'track_id': track_id,
                    'confidence': punch_conf
                })
                fight_score += punch_conf * 0.4

            is_fall = self.fall_detector.detect(bbox, track_id)
            if is_fall:
                events.append({
                    'type': 'fall',
                    'track_id': track_id
                })
                fight_score += 0.3

        interactions = self.interaction_detector.detect(tracked_keypoints, [t['track_id'] for t in tracked_results])
        for inter in interactions:
            if inter['is_contact']:
                events.append({
                    'type': 'interaction',
                    'persons': (inter['person1'], inter['person2']),
                    'contact_parts': inter['contact_parts']
                })
                fight_score += 0.3

        fight_score = min(1.0, fight_score)
        for track in tracked_results:
            track_id = track['track_id']
            if track_id not in self.fighting_tracks:
                self.fighting_tracks[track_id] = 0

            has_event = any(e.get('track_id') == track_id for e in events if e['type'] in ['punch', 'fall'])
            has_interaction = any(track_id in inter.get('persons', ()) for inter in interactions)

            if has_event or has_interaction:
                self.fighting_tracks[track_id] += 1
            else:
                self.fighting_tracks[track_id] = max(0, self.fighting_tracks[track_id] - 1)

        is_fighting = fight_score > 0.5 or any(count >= Config.FIGHT_FRAMES_THRESHOLD for count in self.fighting_tracks.values())
        print(f"is_fighting: {is_fighting},fight_score: {fight_score}")
        self.fight_history.append(is_fighting)
        #print(f"fight_history: {self.fight_history}")
        # 清理不活跃ID，防止内存泄漏
        self._cleanup_inactive(active_ids)
        return {
            'is_fighting': is_fighting,
            'fight_score': fight_score,
            'events': events,
            'tracked_persons': tracked_results,
            'interactions': interactions
        }

    def _cleanup_inactive(self, active_ids):
        """清理所有检测器中的不活跃track_id"""
        self.punch_detector.cleanup(active_ids)
        self.fall_detector.cleanup(active_ids)
        for tid in list(self.fighting_tracks.keys()):
            if tid not in active_ids:
                del self.fighting_tracks[tid]


# ========== 绘图函数 ==========
def draw_keypoints_and_skeleton(image, keypoints):
    img_copy = image.copy()

    if keypoints is None or len(keypoints) < 17:
        return img_copy

    for start_idx, end_idx in SKELETON:
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue

        start_point = keypoints[start_idx][:2]
        end_point = keypoints[end_idx][:2]

        if (start_point[0] > 0 and start_point[1] > 0 and 
            end_point[0] > 0 and end_point[1] > 0):

            start_tuple = (int(start_point[0]), int(start_point[1]))
            end_tuple = (int(end_point[0]), int(end_point[1]))

            if start_idx in [0,1,2,3,4] or end_idx in [0,1,2,3,4]:
                color = COLORS['head']
            elif start_idx in [5,6,7,8,9,10] or end_idx in [5,6,7,8,9,10]:
                color = COLORS['arms']
            elif start_idx in [11,12] or end_idx in [11,12]:
                color = COLORS['body']
            else:
                color = COLORS['legs']

            cv2.line(img_copy, start_tuple, end_tuple, color, 2, cv2.LINE_AA)

    for i, point in enumerate(keypoints):
        if point[0] > 0 and point[1] > 0:
            center = (int(point[0]), int(point[1]))
            color = KEYPOINT_COLORS[i] if i < len(KEYPOINT_COLORS) else (255, 255, 255)
            cv2.circle(img_copy, center, 4, color, -1, cv2.LINE_AA)
            cv2.circle(img_copy, center, 6, (255, 255, 255), 1, cv2.LINE_AA)

    return img_copy


def draw_fighting_info(image, fighting_result, tracked_results):
    img_copy = image.copy()
    try:
        # 绘制跟踪框
        for track in tracked_results:
            x1, y1, x2, y2 = track['bbox']
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, f"ID:{track['track_id']}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制交互连线
        interactions = fighting_result.get('interactions', [])
        for inter in interactions:
            tid1 = inter['person1']
            tid2 = inter['person2']
            # 找到对应bbox中心点
            bbox1 = None
            bbox2 = None
            for t in tracked_results:
                if t['track_id'] == tid1:
                    bbox1 = t['bbox']
                if t['track_id'] == tid2:
                    bbox2 = t['bbox']
            if bbox1 is not None and bbox2 is not None:
                c1 = (int((bbox1[0]+bbox1[2])/2), int((bbox1[1]+bbox1[3])/2))
                c2 = (int((bbox2[0]+bbox2[2])/2), int((bbox2[1]+bbox2[3])/2))
                color = (0, 0, 255) if inter['is_contact'] else (0, 255, 255)
                thickness = 3 if inter['is_contact'] else 1
                cv2.line(img_copy, c1, c2, color, thickness, cv2.LINE_AA)
                mid = ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2)
                cv2.putText(img_copy, f"{inter['distance']:.0f}px", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 打架总提示
        if fighting_result['is_fighting']:
            cv2.putText(img_copy, f"FIGHTING! Score:{fighting_result['fight_score']:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 事件列表
        y_offset = 60
        for event in fighting_result['events']:
            if event['type'] == 'punch':
                msg = f"Punch! ID:{event['track_id']} conf:{event['confidence']:.2f}"
                cv2.putText(img_copy, msg, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 25
            elif event['type'] == 'fall':
                msg = f"Fall! ID:{event['track_id']}"
                cv2.putText(img_copy, msg, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                y_offset += 25
            elif event['type'] == 'interaction':
                p1, p2 = event['persons']
                parts = ','.join(event['contact_parts']) if event['contact_parts'] else 'near'
                msg = f"Interact! ID{p1}<->ID{p2} [{parts}]"
                color = (0, 0, 255) if event['contact_parts'] else (0, 255, 255)
                cv2.putText(img_copy, msg, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
    except Exception as e:
        print(f"绘制信息失败: {e}")

    return img_copy


# ========== 主函数 ==========
def process_video(video_path, det_model_path, pose_model_path='yolov8n-pose.pt'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return

    print("加载检测模型...")
    try:
        det_model = YOLO(det_model_path)
    except Exception as e:
        print(f"加载检测模型失败: {e}")
        return

    print("加载姿态模型...")
    try:
        pose_model = YOLO(pose_model_path)
    except Exception as e:
        print(f"加载姿态模型失败: {e}")
        return

    fighting_detector = FightingDetector(Config.TRACK_CONFIG)

    frame_id = 0
    total_time = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            import time
            t0 = time.time()

            try:
                det_results = det_model(frame, classes=[0], conf=Config.DET_CONF, verbose=False)
            except Exception as e:
                print(f"帧 {frame_id} 检测异常: {e}")
                frame_id += 1
                continue

            detections = []
            keypoints_list = []
            all_dets = det_results[0].boxes.data.cpu().numpy()
    
            for det in all_dets:
                if det[4] < Config.DET_CONF or int(det[5]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, det[:4])
                # 统一为6列格式 [x1, y1, x2, y2, conf, cls]
                detections.append([x1, y1, x2, y2, float(det[4]), int(det[5])])

                h, w = y2 - y1, x2 - x1
                px1 = max(0, x1 - int(w * 0.1))
                py1 = max(0, y1 - int(h * 0.1))
                px2 = min(frame.shape[1], x2 + int(w * 0.1))
                py2 = min(frame.shape[0], y2 + int(h * 0.1))

                person_crop = frame[py1:py2, px1:px2]
                if person_crop.size == 0:
                    keypoints_list.append(None)
                    continue

                try:
                    #  传入裁剪后的图像 应该使用并行加速
                    pose_results = pose_model(person_crop, conf=Config.POSE_CONF, verbose=False)

                    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
                        keypoints = pose_results[0].keypoints.xy.cpu().numpy()[0]
                        keypoints[:, 0] += px1
                        keypoints[:, 1] += py1
                        keypoints_list.append(keypoints)
                    else:
                        keypoints_list.append(None)
                except Exception as e:
                    print(f"帧 {frame_id} 姿态估计异常: {e}")
                    keypoints_list.append(None)
            print(f"帧 {frame_id} 姿态结果: {len(keypoints_list)}, 检测结果: {len(detections)}")
            # 传入原始图像 frame
            fighting_result = fighting_detector.process_frame(detections, keypoints_list, frame)
            result_frame = frame.copy()
            for kp in keypoints_list:
                result_frame = draw_keypoints_and_skeleton(result_frame, kp)

            tracked_results = fighting_result.get('tracked_persons', [])
  
            result_frame = draw_fighting_info(result_frame, fighting_result, tracked_results)

            # 显示FPS
            fps = 1.0 / (time.time() - t0 + 1e-6)
            cv2.putText(result_frame, f"FPS:{fps:.1f}", (frame.shape[1]-120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Fighting Detection", result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"运行时异常: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"处理完成，共 {frame_id} 帧")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多人打架行为检测系统")
    parser.add_argument("--video_path", type=str, required=True, help="输入视频路径")
    parser.add_argument("--det_model", type=str, required=True, 
                       help="YOLO检测模型路径（如 runs/train/weights/best.pt）")
    parser.add_argument("--pose_model", type=str, default="yolov8n-pose.pt",
                       help="YOLO姿态模型路径（默认: yolov8n-pose.pt）")
    args = parser.parse_args()
    process_video(args.video_path, args.det_model, args.pose_model)