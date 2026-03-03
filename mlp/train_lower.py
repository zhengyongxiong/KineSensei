# train_lower.py
# ------------------------------------------------------------
# 适配新的 10-point 下肢特征 (长度自动由 X_lower.npy 推断)
# - 仅在训练集上 fit StandardScaler，避免数据泄漏
# - 数据增强：高斯噪声 + 随机缩放 + 左右镜像
# - Early-Stopping + 最佳模型保存
# ------------------------------------------------------------
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
import joblib

# ========= 可复现随机种子 =========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========= 路径 =========
FEATURE_DIR = "features"
X_PATH = os.path.join(FEATURE_DIR, "X_lower.npy")
Y_PATH = os.path.join(FEATURE_DIR, "y_lower.npy")

# ========= 超参数 =========
HIDDEN_DIM     = 128
BATCH_SIZE     = 64
LR             = 1e-3
EPOCHS         = 100
PATIENCE       = 10          # Early-Stopping
NOISE_STD      = 0.02
SCALE_RANGE    = (0.95, 1.05)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 模型 =========
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

# ========= 数据加载 =========
X = np.load(X_PATH)
y = np.load(Y_PATH)
INPUT_DIM    = X.shape[1]
NUM_CLASSES  = len(np.unique(y))

# ========= 数据增强 =========
def augment(x: np.ndarray) -> np.ndarray:
    """高斯噪声 + 随机缩放 + 概率 0.5 左右镜像 (交换左右关键点)"""
    x_aug = x.copy()

    # 高斯噪声
    x_aug += np.random.normal(0, NOISE_STD, x.shape)

    # 随机缩放
    scale = np.random.uniform(*SCALE_RANGE, (x.shape[0], 1))
    x_aug *= scale

    # 左右镜像：仅对 x 坐标取 1-prob 翻转
    if np.random.rand() < 0.5:
        # 每个关键点都是 (x,y) 连续存储：swap 左(偶)↔右(奇) 的 x 坐标
        # 第 0-9 点：LH RH LK RK LA RA LHEEL RHEEL LFOOT RFOOT
        swap_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        for l, r in swap_pairs:
            # 取对应 (x,y) → 索引 = 2*idx
            lx, rx = 2*l, 2*r
            x_aug[:, [lx, rx]] = x_aug[:, [rx, lx]]
    return x_aug

X_aug = augment(X)
X     = np.vstack([X, X_aug])
y     = np.concatenate([y, y])

# ========= 数据划分 =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# 标准化器仅 fit 训练集
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# ========= DataLoader =========
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                  torch.tensor(y_test, dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ========= 初始化 =========
model = PoseClassifier(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_acc   = 0.0
patience   = 0

# ========= 训练循环 =========
for epoch in range(1, EPOCHS + 1):
    # -------- Train --------
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # -------- Validation --------
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    val_acc = correct / total
    print(f"[LOWER] Epoch {epoch:03d}/{EPOCHS} | "
          f"Loss: {running_loss:.4f} | Val Acc: {val_acc:.3f}")

    # -------- Early-Stopping --------
    if val_acc > best_acc:
        best_acc = val_acc
        patience = 0
        torch.save(model.state_dict(), "mlp_lower_model.pth")
    else:
        patience += 1
        if patience >= PATIENCE:
            print("🥇 Early-Stopping triggered.")
            break

# ========= 报告 =========
model.load_state_dict(torch.load("mlp_lower_model.pth", map_location=DEVICE))
model.eval()
y_pred = []

with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(DEVICE)
        y_pred.extend(model(xb).argmax(dim=1).cpu().numpy())

print("\n=== Classification Report (Test Set) ===")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ========= 保存 StandardScaler =========
joblib.dump(scaler, "scaler_lower.pkl")
print(f"✅ 模型训练完成！最佳验证准确率: {best_acc:.3f}")
