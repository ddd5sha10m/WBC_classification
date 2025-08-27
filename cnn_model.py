# ===================================================================
# PyTorch 版本的 CNN 血球分類腳本
# ===================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F

# --- 1. 設備設定：明確指定使用 Apple Silicon GPU (MPS) ---
print("---")
print("正在檢查與設定設備...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("成功偵測到 MPS 設備！將使用 GPU 進行訓練。")
else:
    device = torch.device("cpu")
    print("警告：未偵測到 MPS 設備。將使用 CPU 進行訓練。")
print("---")


# --- 2. 載入並準備資料 ---
print("開始載入並準備資料...")

# 載入儲存好的 .npy 檔案
PROCESSED_IMAGES_FILE = "/Users/wdwddaniel/Desktop/side_projects/ALL_cells/processed_data/processed_images.npy"
PROCESSED_LABELS_FILE = "/Users/wdwddaniel/Desktop/side_projects/ALL_cells/processed_data/processed_labels.npy"

images_np = np.load(PROCESSED_IMAGES_FILE)
labels_np = np.load(PROCESSED_LABELS_FILE)

# 將資料集切分成訓練集和驗證集
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split( #test_size=0.2：20% 當驗證集。random_state=42：固定隨機種子以利重現（只針對 train_test_split）。
    images_np, labels_np, test_size=0.2, random_state=42, stratify=labels_np #stratify=labels_np：⭐保留原始資料的類別比例到 train/val（重要，避免類別分布偏差）。
)

# 在您的 PyTorch 訓練腳本中加入
from torchvision import transforms

# --- 僅對訓練資料進行增強 ---
# 驗證和測試資料不應該被隨機改變，所以只做正規化
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    # 針對「顏色」
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    # 針對「大小」和視角
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
])

# 對於驗證和測試集，我們只轉換成 Tensor
val_test_transforms = transforms.Compose([
    transforms.ToTensor(),
])
# --- 建立自訂的 Dataset 類別來應用 transforms ---
from torch.utils.data import Dataset

class CellDataset(Dataset):
    def __init__(self, images_np, labels_np, transform=None):
        self.images = images_np
        self.labels = torch.from_numpy(labels_np).long()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 在這裡，每次要取出一筆資料時，都會即時地進行轉換
        if self.transform:
            image = self.transform(image)
            
        return image, label
# 建立訓練集和驗證集的 Dataset
train_dataset = CellDataset(X_train_np, y_train_np, transform=train_transforms)
val_dataset = CellDataset(X_val_np, y_val_np, transform=val_test_transforms) # 驗證集使用簡單的轉換

'''
# 將 NumPy 陣列轉換為 PyTorch Tensors
# 注意 PyTorch 的影像格式是 (N, C, H, W)，所以我們需要用 permute(0, 3, 1, 2) 來交換維度
#transforms.Normalize(mean,std)）—⭐這是非常關鍵，若沒標準化會導致訓練不穩或收斂慢。這段就在標準化影像資料。
X_train = torch.from_numpy(X_train_np).permute(0, 3, 1, 2).float() #permute(0,3,1,2)：把資料從 (N, H, W, C) 變成 (N, C, H, W)（PyTorch 要的格式）。
y_train = torch.from_numpy(y_train_np).long() #.long()：把 label 轉成整數型（CrossEntropy 要的型別）。
X_val = torch.from_numpy(X_val_np).permute(0, 3, 1, 2).float()
y_val = torch.from_numpy(y_val_np).long()

# 建立 PyTorch Dataset 和 DataLoader
# DataLoader 可以幫助我們自動產生批次 (batch) 資料
train_dataset = TensorDataset(X_train, y_train)#TensorDataset：把 x,y 綁在一起供 DataLoader 抽取。
val_dataset = TensorDataset(X_val, y_val)
'''
BATCH_SIZE = 64 # 您可以根據您的記憶體大小調整
#shuffle=True：⭐訓練集要打亂（避免序列相關 bias）。驗證集通常 shuffle=False。
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) #DataLoader：負責分批、打亂（shuffle），把資料送進 model。
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)#BATCH_SIZE：控制每次前傳/反傳的樣本數，會影響 GPU/記憶體與訓練穩定性（batch size 太大/太小都會影響效果）。
#可選強化項：num_workers、pin_memory（加速 I/O）、drop_last（在需要穩定 batch 大小時）。



print("資料準備完成！")
print(f"訓練集批次數量: {len(train_loader)}")
print(f"驗證集批次數量: {len(val_loader)}")
print("---")

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001) # 加入 Batch Normalization，有助於穩定訓練

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# --- 3. 建立 CNN 模型 ---
# 在 PyTorch 中，我們通常建立一個 class 來定義模型結構
'''
class CellCNN(nn.Module):
    def __init__(self, num_classes):
        
        super(CellCNN, self).__init__()
        # Keras: Conv2D(32, (3, 3), activation='relu', input_shape=...)
        # PyTorch: nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #nn.Conv2d(3,32,3,padding=1)：3 通道輸入（RGB）→ 32 個濾波器； kernel=3，padding=1 保持空間大小不變。
            nn.ReLU(),#nn.ReLU()：非線性 activation。
            nn.MaxPool2d(kernel_size=2, stride=2),#每次將 H、W 各除以 2（下採樣）。三個 block 意味著空間尺寸會被 /2 三次。
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        
        # Keras: Flatten() + Dense()
        # 經過三次 MaxPooling (128 -> 64 -> 32 -> 16), 影像尺寸變為 16x16
        # 所以展平後的特徵數量為 128 (channels) * 16 * 16
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

'''
血球的特徵大小各不相同：

小區域特徵：細胞質中的細微顆粒 (Granularity)。

中區域特徵：核仁 (Nucleoli) 的輪廓、細胞核的分葉 (Lobulation) 邊緣。

大區域特徵：整個細胞核與細胞質的比例 (N:C ratio)、細胞的整體形狀和大小。

要讓模型同時關注到這些不同尺度的特徵，最經典、最直接的方法就是引入 Inception 模組 的設計思想。

我們可以建立一個自訂的 InceptionBlock 類別，然後用它來替換您模型中的原有區塊。

第一步：定義 InceptionBlock 類別

這個類別會包含 1x1, 3x3, 5x5 的卷積分支。
### 核心方法：Inception 模組 (Inception Module)
Inception 模組（源自 GoogLeNet）的核心思想是：與其讓網路自己選擇用多大的卷積核（kernel）最好，不如**「我全都要！」。它會同時使用不同大小的卷積核**（例如 1x1, 3x3, 5x5）來平行處理輸入的特徵，然後將所有分支的結果合併（Concatenate）起來，傳給下一層。

這就像一個醫療團隊在看同一張抹片：

1x1 卷積核：像一位專注於像素級細節的技術員。

3x3 卷積核：像一位關注局部特徵（如顆粒）的醫檢師。

5x5 卷積核：像一位觀察整體結構（如細胞核形狀）的病理醫師。

池化層 (Pooling)：提供一個高度概括的資訊。
'''
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
        return torch.cat(outputs, 1) # 1 是通道維度


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
            nn.Dropout(0.7), # 保留 Dropout 防止過擬合
            nn.Linear(128, num_classes) # 最終輸出到分類數量
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 實例化模型並移至 MPS 設備
num_classes = len(np.unique(labels_np))
model = CellCNN(num_classes).to(device)

# 為了看到模型結構，可以安裝 torchinfo: pip install torchinfo
# from torchinfo import summary
# summary(model, input_size=(BATCH_SIZE, 3, 128, 128))


# --- 4. 定義損失函數與優化器 ---
# PyTorch 的 CrossEntropyLoss = LogSoftmax + NLLLoss，所以模型輸出層不需要加 Softmax
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=5, gamma=0.5) # 每 5 個 epoch 將學習率減半
# --- 5. 訓練與驗證迴圈 ---
best_val_loss = float('inf')
epochs_no_improve = 0
patience = 5
print("開始模型訓練...")
NUM_EPOCHS = 20 # 您可以自行設定訓練的週期數

for epoch in range(NUM_EPOCHS):
    # --- 訓練模式 ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    # 使用 tqdm 顯示進度條
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [訓練中]")
    for inputs, labels in train_progress_bar:
        # 將資料移至 MPS 設備
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向傳播與優化
        loss.backward()
        optimizer.step()
        
        # 計算統計數據
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        # 更新進度條
        train_progress_bar.set_postfix(loss=loss.item())

    train_accuracy = 100 * correct_train / total_train
    train_loss = running_loss / len(train_loader)

    # --- 驗證模式 ---
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad(): # 在驗證時，我們不需要計算梯度
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [驗證中]")
        for inputs, labels in val_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    val_loss = running_val_loss / len(val_loader)
    # 更新學習率調度器
    scheduler.step()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # 儲存表現最好的模型
        torch.save(model.state_dict(), '/Users/wdwddaniel/Desktop/side_projects/ALL_cells/model/cell_cnn_model4.pth')
        print(f"驗證損失改善 ({best_val_loss:.4f})，儲存模型至您指定的位置")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"連續 {patience} 個 epochs 驗證損失未改善，提早停止訓練。")
            break
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"訓練損失: {train_loss:.4f}, 訓練準確率: {train_accuracy:.2f}% | "
          f"驗證損失: {val_loss:.4f}, 驗證準確率: {val_accuracy:.2f}%")
    

print("---")
print("模型訓練完成！")

# 之後您可以加上儲存模型的程式碼