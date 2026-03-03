import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib

# === 超参数配置 ===
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_DIM = 128
NUM_CLASSES = 9

# === 数据增强函数（加噪声）===
def augment(X):
    noise = np.random.normal(0, 0.02, X.shape)
    return X + noise

# === 加载与预处理数据（融合后的83维特征）===
X = np.load("X.npy")     # shape: (N, 83)
y = np.load("y.npy")     # shape: (N, )
INPUT_DIM = X.shape[1]   # 自动设置为实际维度

# === 数据增强 ===
X_aug = augment(X)
X = np.vstack([X, X_aug])
y = np.concatenate([y, y])

# === 标准化 ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === 数据集划分 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# === PyTorch 数据加载 ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# === MLP 模型结构（带 BN 和 Dropout）===
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

model = PoseClassifier(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)

# === 损失函数与优化器 ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === 模型训练 ===
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # === 验证 ===
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = torch.argmax(model(xb), dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Val Acc: {acc:.3f}")

# === 测试集评估 ===
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = torch.argmax(model(xb), dim=1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(yb.cpu().numpy())

print("\n📊 分类报告:")
print(classification_report(y_true, y_pred, digits=2))
print("🧩 混淆矩阵:")
print(confusion_matrix(y_true, y_pred))

# === 保存模型与标准化器 ===
torch.save(model.state_dict(), "mlp_pose_model.pth")
joblib.dump(scaler, "scaler.pkl")
