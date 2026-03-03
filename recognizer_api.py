import numpy as np
import joblib
import torch
import torch.nn as nn
import json
import mediapipe as mp
import cv2
from typing import List

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

# ========= 特征函数 =========
def _distances(pts):
    return np.linalg.norm(pts[:-1] - pts[1:], axis=1)

def _angles(pts):
    vec_a = pts[:-2] - pts[1:-1]
    vec_b = pts[2:] - pts[1:-1]
    cos = np.sum(vec_a * vec_b, axis=1) / (
        np.linalg.norm(vec_a, axis=1) * np.linalg.norm(vec_b, axis=1) + 1e-6
    )
    return np.arccos(np.clip(cos, -1.0, 1.0))

def extract_upper_features(pts):  # (12, 2)
    return np.concatenate([pts.flatten(), _distances(pts), _angles(pts)])

def lower_enhanced_features(pts):  # (10, 2)
    d = lambda i, j: np.linalg.norm(pts[i] - pts[j])
    ang = lambda p0, p1, p2: np.arccos(np.clip(
        np.dot(pts[p0] - pts[p1], pts[p2] - pts[p1]) /
        (np.linalg.norm(pts[p0] - pts[p1]) * np.linalg.norm(pts[p2] - pts[p1]) + 1e-6),
        -1.0, 1.0))
    left_thigh = d(0, 2); right_thigh = d(1, 3)
    left_shin = d(2, 4); right_shin = d(3, 5)
    left_leg = d(0, 4); right_leg = d(1, 5)
    hip_sep = d(0, 1); knee_sep = d(2, 3)
    ankle_sep = d(4, 5); foot_sep = d(8, 9)
    left_knee_ang = ang(0, 2, 4)
    right_knee_ang = ang(1, 3, 5)
    hip_tilt_ang = ang(1, 0, 2)
    hip_center = (pts[0] + pts[1]) / 2
    foot_center = (pts[8] + pts[9]) / 2
    h_bias = abs(hip_center[0] - foot_center[0])
    v_bias = abs(hip_center[1] - foot_center[1])
    return np.array([
        left_thigh, right_thigh, left_shin, right_shin,
        left_leg, right_leg, hip_sep, knee_sep, ankle_sep, foot_sep,
        left_knee_ang, right_knee_ang, hip_tilt_ang, h_bias, v_bias
    ])

def extract_lower_features(pts):
    return np.concatenate([pts.flatten(), lower_enhanced_features(pts)])

# ========= 标签映射 =========
with open("mlp/label_map_upper.json", encoding="utf-8") as f:
    id2label_upper = {v: k for k, v in json.load(f).items()}
with open("mlp/label_map_lower.json", encoding="utf-8") as f:
    id2label_lower = {v: k for k, v in json.load(f).items()}

# ========= 加载模型与标准化器 =========
upper_dim = np.load("mlp/X_upper.npy", mmap_mode="r").shape[1]
lower_dim = np.load("mlp/X_lower.npy", mmap_mode="r").shape[1]

upper_model = PoseClassifier(upper_dim, 128, len(id2label_upper))
upper_model.load_state_dict(torch.load("mlp/mlp_upper_model.pth", map_location="cpu"))
upper_model.eval()
scaler_upper = joblib.load("mlp/scaler_upper.pkl")

lower_model = PoseClassifier(lower_dim, 128, len(id2label_lower))
lower_model.load_state_dict(torch.load("mlp/mlp_lower_model.pth", map_location="cpu"))
lower_model.eval()
scaler_lower = joblib.load("mlp/scaler_lower.pkl")

# ========= MediaPipe 初始化 =========
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# ========= 主接口函数 =========
def classify_pose_from_image(image: np.ndarray) -> dict:
    """图像输入，返回 {"upper": label, "lower": label}"""
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)

        if not results.pose_landmarks:
            return {"upper": "NE_Upper", "lower": "NE_Lower"}

        keypoints = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]
        return classify_pose_from_keypoints(keypoints)
    except Exception as e:
        print("❌ classify_pose_from_image 错误：", e)
        return {"upper": "NE_Upper", "lower": "NE_Lower"}

def classify_pose_from_keypoints(keypoints: List[List[float]]) -> dict:
    """关键点输入，返回 {"upper": label, "lower": label}"""
    result = {"upper": "NE_Upper", "lower": "NE_Lower"}
    try:
        pts = np.array(keypoints)
        up_pts = pts[11:23]
        if up_pts.shape != (12, 2):
            raise ValueError("upper 关键点数量不正确")
        feat = extract_upper_features(up_pts)
        feat = scaler_upper.transform([feat])
        pred = upper_model(torch.tensor(feat, dtype=torch.float32)).argmax(1).item()
        result["upper"] = id2label_upper.get(pred, f"U:{pred}")
    except Exception as e:
        print("⚠️ 上肢识别失败：", e)

    try:
        lo_pts = pts[23:33]
        if lo_pts.shape != (10, 2):
            raise ValueError("lower 关键点数量不正确")
        feat = extract_lower_features(lo_pts)
        feat = scaler_lower.transform([feat])
        pred = lower_model(torch.tensor(feat, dtype=torch.float32)).argmax(1).item()
        result["lower"] = id2label_lower.get(pred, f"L:{pred}")
    except Exception as e:
        print("⚠️ 下肢识别失败：", e)

    return result

def classify_confidence_from_keypoints(keypoints: List[List[float]], target_action: str, part: str = "upper") -> dict:
    """
    返回目标动作的置信度和是否通过
    part: "upper" 或 "lower"
    """
    try:
        pts = np.array(keypoints)
        if part == "upper":
            sub_pts = pts[11:23]
            if sub_pts.shape != (12, 2):
                raise ValueError("upper 关键点数量不正确")
            feat = extract_upper_features(sub_pts)
            feat = scaler_upper.transform([feat])
            logits = upper_model(torch.tensor(feat, dtype=torch.float32)).detach().numpy().flatten()
            label2id = {v: k for k, v in id2label_upper.items()}
        else:  # part == "lower"
            sub_pts = pts[23:33]
            if sub_pts.shape != (10, 2):
                raise ValueError("lower 关键点数量不正确")
            feat = extract_lower_features(sub_pts)
            feat = scaler_lower.transform([feat])
            logits = lower_model(torch.tensor(feat, dtype=torch.float32)).detach().numpy().flatten()
            label2id = {v: k for k, v in id2label_lower.items()}

        if target_action not in label2id:
            raise ValueError(f"未找到动作标签：{target_action}")

        prob = torch.softmax(torch.tensor(logits), dim=0).numpy()
        confidence = float(prob[label2id[target_action]])
        return {
            "confidence": confidence,
            "pass": confidence > 0.85  # ✅ 可调节阈值
        }

    except Exception as e:
        print("⚠️ classify_confidence_from_keypoints 错误：", e)
        return {
            "confidence": 0.0,
            "pass": False
        }
