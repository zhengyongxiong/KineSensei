# recognizer_dual.py
# ------------------------------------------------------------
# 实时上肢 + 下肢姿态分类
#   · upper: 12 pts → 45-dim 特征
#   · lower: 10 pts → 35-dim 特征
# ------------------------------------------------------------
import cv2
import numpy as np
import torch
import json
import joblib
import mediapipe as mp
from collections import deque
from statistics import mode
import torch.nn as nn

# ========= 模型结构 =========
class PoseClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ========= 上肢特征函数 =========
def _distances(pts):
    return np.linalg.norm(pts[:-1] - pts[1:], axis=1)  # 相邻距离

def _angles(pts):
    vec_a = pts[:-2] - pts[1:-1]
    vec_b = pts[2:]  - pts[1:-1]
    cos = np.sum(vec_a * vec_b, axis=1) / (
        np.linalg.norm(vec_a, axis=1) * np.linalg.norm(vec_b, axis=1) + 1e-6
    )
    return np.arccos(np.clip(cos, -1.0, 1.0))

def extract_upper_features(pts):        # (12,2)
    return np.concatenate([pts.flatten(), _distances(pts), _angles(pts)])

# ========= 下肢特征函数 =========
def lower_enhanced_features(pts):       # (10,2)
    d   = lambda i,j: np.linalg.norm(pts[i]-pts[j])
    ang = lambda p0,p1,p2: np.arccos(
        np.clip(
            np.dot(pts[p0]-pts[p1], pts[p2]-pts[p1]) /
            (np.linalg.norm(pts[p0]-pts[p1]) *
             np.linalg.norm(pts[p2]-pts[p1]) + 1e-6),
            -1.0, 1.0))

    left_thigh   = d(0, 2); right_thigh  = d(1, 3)
    left_shin    = d(2, 4); right_shin   = d(3, 5)
    left_leg     = d(0, 4); right_leg    = d(1, 5)

    hip_sep      = d(0, 1); knee_sep  = d(2, 3)
    ankle_sep    = d(4, 5); foot_sep  = d(8, 9)

    left_knee_ang  = ang(0,2,4)
    right_knee_ang = ang(1,3,5)
    hip_tilt_ang   = ang(1,0,2)

    hip_center  = (pts[0] + pts[1]) / 2
    foot_center = (pts[8] + pts[9]) / 2
    h_bias, v_bias = abs(hip_center[0]-foot_center[0]), abs(hip_center[1]-foot_center[1])

    return np.array([
        left_thigh, right_thigh, left_shin, right_shin,
        left_leg, right_leg,
        hip_sep, knee_sep, ankle_sep, foot_sep,
        left_knee_ang, right_knee_ang, hip_tilt_ang,
        h_bias, v_bias
    ])

def extract_lower_features(pts):
    return np.concatenate([pts.flatten(), lower_enhanced_features(pts)])

# ========= 读取类别映射 =========
with open("features/label_map_upper.json", encoding="utf-8") as f:
    id2label_upper = {v: k for k, v in json.load(f).items()}
with open("features/label_map_lower.json", encoding="utf-8") as f:
    id2label_lower = {v: k for k, v in json.load(f).items()}

NUM_CLASSES_UP = len(id2label_upper)     # 3
NUM_CLASSES_LO = len(id2label_lower)

# ========= 输入维度 =========
upper_dim = np.load("features/X_upper.npy", mmap_mode="r").shape[1]  # 45
lower_dim = np.load("features/X_lower.npy", mmap_mode="r").shape[1]  # 35

# ========= 加载模型 & Scaler =========
upper_model = PoseClassifier(upper_dim, 128, NUM_CLASSES_UP)
upper_model.load_state_dict(torch.load("mlp_upper_model.pth", map_location="cpu"))
upper_model.eval()
scaler_upper = joblib.load("scaler_upper.pkl")

lower_model = PoseClassifier(lower_dim, 128, NUM_CLASSES_LO)
lower_model.load_state_dict(torch.load("mlp_lower_model.pth", map_location="cpu"))
lower_model.eval()
scaler_lower = joblib.load("scaler_lower.pkl")

# ========= MediaPipe =========
mp_pose  = mp.solutions.pose
pose     = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawer   = mp.solutions.drawing_utils

# ========= 投票缓冲 =========
queue_upper = deque(maxlen=5)
queue_lower = deque(maxlen=5)

# ========= 摄像头 =========
cap = cv2.VideoCapture(1)    # 若外接摄像头请改成 1
LOWER_VIS_TH = 0.3

print("⌛ Camera ready…  Press 'q' to quit.")
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    upper_out, lower_out = "NE_Upper", "NE_Lower"
    if res.pose_landmarks:
        lm  = res.pose_landmarks.landmark
        pts = np.array([[p.x, p.y] for p in lm])   # (33,2)

        # ---------- 上肢 ----------
        try:
            up_pts = pts[11:23]
            feat   = scaler_upper.transform([extract_upper_features(up_pts)])
            pred   = upper_model(torch.tensor(feat, dtype=torch.float32)).argmax(1).item()
            queue_upper.append(pred)
            upper_out = id2label_upper.get(mode(queue_upper), f"U:{pred}")
        except Exception:
            pass

        # ---------- 下肢 ----------
        try:
            low_idx = list(range(23,33))
            if all(lm[i].visibility > LOWER_VIS_TH for i in low_idx):
                lo_pts = pts[23:33]
                feat   = scaler_lower.transform([extract_lower_features(lo_pts)])
                pred   = lower_model(torch.tensor(feat, dtype=torch.float32)).argmax(1).item()
                queue_lower.append(pred)
                lower_out = id2label_lower.get(mode(queue_lower), f"L:{pred}")
            else:
                lower_out = "NE_Lower(vis)"
        except Exception:
            pass

        drawer.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # ---------- 显示 ----------
    cv2.putText(img, f"{upper_out} + {lower_out}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    cv2.imshow("🕺 Pose Classifier", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
