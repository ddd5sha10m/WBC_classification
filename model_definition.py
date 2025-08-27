import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. 設備設定 ---
print("---")
print("正在檢查與設定設備...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("成功偵測到 MPS 設備！")
else:
    device = torch.device("cpu")
    print("警告：未偵測到 MPS 設備。將使用 CPU。")
print("---")


# --- 2. 重新定義模型架構 ---
# !!! 警告：這裡的模型類別，必須和您訓練時使用的 CellCNN 類別一模一樣 !!!
class CellCNN(nn.Module):
    def __init__(self, num_classes):
        super(CellCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32,64 ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64 , 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32 , 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes) # 輸出層，CrossEntropyLoss 會自動處理 softmax
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x