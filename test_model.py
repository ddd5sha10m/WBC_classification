# ===================================================================
# PyTorch 模型測試與評估腳本
# ===================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
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
'''
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
'''

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001) # 加入 Batch Normalization，有助於穩定訓練

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionBlock, self).__init__()
        # 1x1 卷積分支 (最小尺度)
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        # 3x3 卷積分支 (中等尺度)
        # 我們先用一個 1x1 卷積來降維，減少計算量
        self.branch3x3_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)

        # 5x5 卷積分支 (較大尺度)
        self.branch5x5_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(32, 64, kernel_size=5, padding=2) # padding=(kernel_size-1)/2

        # Max Pooling 分支
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)
        self.branch_pool_conv = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        
    
    def forward(self, x):
        # 讓輸入 x 平行地流過所有分支
        branch1x1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch_pool = self.branch_pool_conv(x)
        branch_pool = self.branch_pool(branch_pool)

        # 將所有分支的結果在通道維度上合併
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)
    
class CellCNN(nn.Module):
    def __init__(self, num_classes):
        super(CellCNN, self).__init__()
        
        # --- 特徵提取層 (使用 Inception 模組) ---
        self.features = nn.Sequential(
            # 初始卷積層，先做初步的特徵提取和下採樣
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # --- 使用我們自訂的 Inception 模組 ---
            # 輸入通道數 = 64
            # 輸出通道數 = 64(1x1) + 96(3x3) + 64(5x5) + 32(pool) = 256
            InceptionBlock(64, pool_features=32),
            
            # --- 再接一個 Inception 模組，處理更複雜的特徵 ---
            # 輸入通道數 = 256
            # 輸出通道數 = 64 + 96 + 64 + 64 = 288
            InceptionBlock(256, pool_features=64),
            
            # 全域平均池化，取代傳統的 MaxPool2d，可以更好地總結特徵
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # --- 分類器 ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 這裡的輸入維度必須是 features 層輸出的通道數，也就是 288
            nn.Linear(288, 128), # 從 288 維特徵降維到 128
            nn.ReLU(),
            nn.Dropout(0.5), # 保留 Dropout 防止過擬合
            nn.Linear(128, num_classes) # 最終輸出到分類數量
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 3. 載入資料與模型 ---
print("開始載入資料與模型...")

# 載入測試資料
TEST_IMAGES_FILE = "/Users/wdwddaniel/Desktop/side_projects/ALL_cells/processed_data/processed_images.npy"
TEST_LABELS_FILE = "/Users/wdwddaniel/Desktop/side_projects/ALL_cells/processed_data/processed_labels.npy"

images_np = np.load(TEST_IMAGES_FILE)
labels_np = np.load(TEST_LABELS_FILE)
# --- 新增的偵錯程式碼 ---
print("\n--- 偵錯資訊 ---")
print(f"載入的影像陣列 (images_np) 形狀: {images_np.shape}")
print(f"載入的標籤陣列 (labels_np) 形狀: {labels_np.shape}")

# 從形狀中取得數量
num_images = images_np.shape[0] if len(images_np.shape) > 0 else 0
num_labels = labels_np.shape[0] if len(labels_np.shape) > 0 else 0
print(f"影像數量: {num_images}")
print(f"標籤數量: {num_labels}")

if num_images != num_labels:
    print("\n🔥 錯誤：影像和標籤的數量不匹配！這就是導致程式崩潰的原因。")
    print("請檢查您用來產生這兩個 .npy 檔案的腳本，並重新產生測試資料。")
else:
    print("\n✅ 數量匹配。")
print("----------------\n")
# --- 偵錯結束 ---
# 準備 PyTorch 資料
# 交換維度以符合 PyTorch (N, C, H, W) 格式
X_test = torch.from_numpy(images_np).permute(0, 3, 1, 2).float()
y_test = torch.from_numpy(labels_np).long()

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 載入訓練好的模型
# 假設您有15個類別，並且模型檔名為 cell_cnn_model.pth
NUM_CLASSES = 15 
MODEL_PATH = "/Users/wdwddaniel/Desktop/side_projects/ALL_cells/model/cell_cnn_model3.pth" # <<< 請務必確認您的模型檔案路徑與名稱

model = CellCNN(NUM_CLASSES).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"成功從 '{MODEL_PATH}' 載入模型權重。")
except FileNotFoundError:
    print(f"錯誤：找不到模型檔案 '{MODEL_PATH}'！請確認路徑和檔名是否正確。")
    exit() # 如果找不到模型，就結束程式
print("---")


# --- 4. 執行模型評估 ---
print("開始在測試集上評估模型...")
model.eval()  # 設定為評估模式
all_preds = []
all_labels = []

with torch.no_grad(): # 在評估時，我們不需要計算梯度
    for inputs, labels in tqdm(test_loader, desc="正在預測"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1) # 取得機率最高的類別作為預測結果
        
        # 將這批次的預測和真實標籤收集起來
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
print("評估完成！")
print("---")


# --- 5. 顯示測試結果 ---
print("測試結果分析：")

# 定義類別名稱，用於報告和圖表
# 這裡我們假設標籤是 0 到 14 的數字，對應到您之前的細胞類型
CELL_TYPES = ['BAS', 'EBO', 'EOS', 'KSC', 'LYA', 'LYT', 'MMZ', 'MOB', 'MON', 'MYB', 'MYO', 'NGB', 'NGS', 'PMB', 'PMO']

# 1. 分類報告 (Classification Report)
# 修正後的程式碼
# 我們加入 labels=list(range(NUM_CLASSES)) 來告訴報告我們總共有 15 個類別
report = classification_report(all_labels, all_preds, target_names=CELL_TYPES, labels=list(range(NUM_CLASSES)))
print("\n[分類報告]\n")
print(report)

# 2. 混淆矩陣 (Confusion Matrix)
print("\n[混淆矩陣]\n")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CELL_TYPES, yticklabels=CELL_TYPES)
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.tight_layout()
plt.show()

print("\n所有分析報告已顯示。")
