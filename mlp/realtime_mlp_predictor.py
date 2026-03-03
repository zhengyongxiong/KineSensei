import cv2
import numpy as np
import torch
import json
import mediapipe as mp
import torch.nn as nn
import joblib
from collections import deque
from statistics import mode

# ===== 几何特征计算模块 =====
def compute_distances(points):
    def dist(p1, p2):
        return np.linalg.norm(points[p1] - points[p2])
    pairs = [
        (11, 12), (23, 24),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
        (15, 16), (27, 28)
    ]
    return np.array([dist(i, j) for i, j in pairs])

def compute_angle(p0, p1, p2):
    a = points[p0] - points[p1]
    b = points[p2] - points[p1]
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

def compute_angles(points):
    triplets = [
        (11, 13, 15),
        (12, 14, 16),
        (23, 25, 27),
        (24, 26, 28)
    ]
    return np.array([compute_angle(i, j, k) for i, j, k in triplets])

def compute_ratios(points):
    upper_len = np.linalg.norm(points[11] - points[13]) + np.linalg.norm(points[13] - points[15]) \
              + np.linalg.norm(points[12] - points[14]) + np.linalg.norm(points[14] - points[16])
    lower_len = np.linalg.norm(points[23] - points[25]) + np.linalg.norm(points[25] - points[27]) \
              + np.linalg.norm(points[24] - points[26]) + np.linalg.norm(points[26] - points[28])
    return np.array([upper_len / (lower_len + 1e-6)])

# ===== 模型定义 =====
class PoseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(PoseClassifier, self).__init__()
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

# ===== 加载模型和 scaler =====
INPUT_DIM = 83
HIDDEN_DIM = 128
NUM_CLASSES = 9

model = PoseClassifier(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
model.load_state_dict(torch.load("mlp_pose_model.pth", map_location="cpu"))
model.eval()

scaler = joblib.load("scaler.pkl")
with open("label_map.json", "r", encoding="utf-8") as f:
    id2label = {int(v): k for k, v in json.load(f).items()}

# ===== 初始化 MediaPipe =====
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# ===== 摄像头推理循环 =====
cap = cv2.VideoCapture(1)
smooth_queue = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取摄像头画面")
        break

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    label = "未检测"
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        points = np.array([[pt.x, pt.y] for pt in lm[:33]])

        if points.shape == (33, 2):
            # === 构造全量特征（83维）===
            flat_coords = points.flatten()
            dist_feat = compute_distances(points)
            angle_feat = compute_angles(points)
            ratio_feat = compute_ratios(points)

            feature = np.concatenate([flat_coords, dist_feat, angle_feat, ratio_feat])
            feature = np.clip(feature, 0.0, 1.0)  # 安全裁剪
            feature_scaled = scaler.transform([feature])
            x_tensor = torch.tensor(feature_scaled, dtype=torch.float32)

            with torch.no_grad():
                logits = model(x_tensor)
                pred_id = torch.argmax(logits, dim=1).item()
                smooth_queue.append(pred_id)
                label = id2label.get(mode(smooth_queue), f"ID {pred_id}")
        else:
            label = "关键点不足"

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 显示文字
    cv2.putText(img, f"动作: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("📷 实时舞蹈识别", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
