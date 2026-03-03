# extract_pose_features.py
# ------------------------------------------------------------
# 统一：下肢使用 **10 个关键点** (23–32)；训练 / 推理保持一致  
# Features:
#   ├─ upper : flat + 相邻距离 + 相邻夹角
#   └─ lower : flat + 几何增强(长度、间距、角度、偏移)
# ------------------------------------------------------------
import cv2
import mediapipe as mp
import numpy as np
import os
import json

# ========= 路径与参数 =========
VIDEO_ROOT   = "videos"      # videos/upper/*.mp4, videos/lower/*.mp4
OUTPUT_ROOT  = "features"    # 保存 *.npy / *.json
SKIP_RATE    = 1             # 每隔多少帧抽 1 帧

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ========= 关键点索引 =========
OUTPUT_DICT = {
    "upper": {
        "keypoint_idx": list(range(11, 23))          # 11–22 共 12 个点
    },
    "lower": {
        "keypoint_idx": list(range(23, 33))          # 23–32 共 10 个点
        # 顺序: 0 LH 1 RH 2 LK 3 RK 4 LA 5 RA 6 LHEEL 7 RHEEL 8 LFOOT 9 RFOOT
    }
}

# ========= 通用特征工具 =========
def compute_distances(points):
    """相邻点欧氏距离"""
    return np.array(
        [np.linalg.norm(points[i] - points[i + 1]) for i in range(len(points) - 1)]
    )

def compute_angles(points):
    """相邻三点夹角 (弧度)"""
    angles = []
    for i in range(len(points) - 2):
        a, b = points[i] - points[i + 1], points[i + 2] - points[i + 1]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        angles.append(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.array(angles)

# ========= 下肢增强特征 =========
def lower_enhanced_features(points):
    """
    points: shape (10, 2) – LH RH LK RK LA RA LHEEL RHEEL LFOOT RFOOT
    返回长度 / 间距 / 角度 / 偏移 共 15 维
    """
    d = lambda i, j: np.linalg.norm(points[i] - points[j])
    ang = lambda p0, p1, p2: np.arccos(
        np.clip(
            np.dot(points[p0] - points[p1], points[p2] - points[p1]) /
            (np.linalg.norm(points[p0] - points[p1]) *
             np.linalg.norm(points[p2] - points[p1]) + 1e-6),
            -1.0, 1.0
        )
    )

    # --- 长度 ---
    left_thigh   = d(0, 2)
    right_thigh  = d(1, 3)
    left_shin    = d(2, 4)
    right_shin   = d(3, 5)
    # 整条腿
    left_leg     = d(0, 4)
    right_leg    = d(1, 5)

    # --- 间距 ---
    hip_sep      = d(0, 1)
    knee_sep     = d(2, 3)
    ankle_sep    = d(4, 5)
    foot_sep     = d(8, 9)

    # --- 角度 ---
    left_knee_angle  = ang(0, 2, 4)   # LH–LK–LA
    right_knee_angle = ang(1, 3, 5)   # RH–RK–RA
    hip_tilt_angle   = ang(1, 0, 2)   # RH–LH–LK  (骨盆倾斜)

    # --- 偏移 ---
    hip_center  = (points[0] + points[1]) / 2
    foot_center = (points[8] + points[9]) / 2
    h_bias = abs(hip_center[0] - foot_center[0])
    v_bias = abs(hip_center[1] - foot_center[1])

    return np.array([
        left_thigh, right_thigh, left_shin, right_shin,
        left_leg, right_leg,
        hip_sep, knee_sep, ankle_sep, foot_sep,
        left_knee_angle, right_knee_angle, hip_tilt_angle,
        h_bias, v_bias
    ])

# ========= 初始化 MediaPipe =========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========= 主循环：upper / lower =========
for mode in ["upper", "lower"]:
    X_data, y_data = [], []
    label2id, next_id = {}, 0
    part_folder = os.path.join(VIDEO_ROOT, mode)

    print(f"\n🎯 处理 {mode.upper()} 文件夹 …")
    for filename in os.listdir(part_folder):
        if not filename.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(part_folder, filename)
        label_name = os.path.splitext(filename)[0]

        if label_name not in label2id:
            label2id[label_name] = next_id
            next_id += 1

        print(f"📂 {filename}  →  标签: {label_name}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开: {video_path}")
            continue

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % SKIP_RATE:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                all_pts = np.array([[p.x, p.y] for p in lm])

                kp_idx = OUTPUT_DICT[mode]["keypoint_idx"]
                pts = all_pts[kp_idx]

                # 过滤异常
                if np.any(np.isnan(pts)):
                    frame_idx += 1
                    continue

                # === 构造特征 ===
                if mode == "upper":
                    feat = np.concatenate([pts.flatten(),
                                           compute_distances(pts),
                                           compute_angles(pts)])
                else:  # lower
                    feat = np.concatenate([pts.flatten(),
                                           lower_enhanced_features(pts)])

                X_data.append(feat)
                y_data.append(label2id[label_name])

            frame_idx += 1
        cap.release()

    # === 保存 ===
    np.save(os.path.join(OUTPUT_ROOT, f"X_{mode}.npy"), np.array(X_data))
    np.save(os.path.join(OUTPUT_ROOT, f"y_{mode}.npy"), np.array(y_data))
    with open(os.path.join(OUTPUT_ROOT, f"label_map_{mode}.json"),
              "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)

    print(f"✅ {mode.upper()} 提取完毕  —  样本数: {len(X_data)}  类别: {len(label2id)}")

pose.close()
