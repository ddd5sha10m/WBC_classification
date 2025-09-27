AI 白血球細胞分類專案 (WBC Classification Project)
這是一個使用深度學習技術對白血球（WBC）影像進行分類的專案。專案採用 PyTorch 框架，從資料前處理、模型訓練、效能評估到建立一個即時預測的網頁應用，提供了一個完整的解決方案。

專案特色
模組化腳本: 將資料處理、模型訓練、模型評估和 Web 應用程式等功能拆分成獨立的 Python 腳本，結構清晰，易於維護。

多模型架構探索:

基礎 CNN 模型: 專案核心是一個自訂的卷積神經網路（CNN），作為效能的基準線。

進階模型探索: 專案中也包含了對更複雜模型架構的初步探索，例如受 Inception 啟發的模組，旨在捕捉不同尺度的影像特徵，展現了未來優化的潛力。

資料增強: 在訓練過程中應用多種資料增強技術（如旋轉、翻轉、顏色抖動），以提高模型的泛化能力並減少過擬合。

即時預測介面: 透過 Flask-SocketIO 建立了一個簡單的網頁介面，使用者可以上傳白血球圖片，模型會進行即時分類並回傳結果。

詳細的效能評估: 提供完整的模型評估腳本，可產生分類報告（Classification Report）與混淆矩陣（Confusion Matrix），幫助分析模型在各類別上的表現。

##專案結構
以下是本專案主要檔案的功能說明：
├── get_img.py             # 1. 影像前處理：讀取 TIFF 影像，調整大小並儲存為 NumPy 檔案。
├── cnn_model.py           # 2. 模型訓練：定義模型結構，載入前處理後的資料進行訓練與驗證。
├── test_model.py          # 3. 模型評估：載入訓練好的模型，在測試集上進行預測並產生評估報告。
├── server.py              # 4. 網頁應用：啟動 Flask 伺服器，提供即時影像分類的 API。
├── model_definition.py    # 輔助腳本：定義基礎 CNN 模型結構，供 server.py 和 test_model.py 匯入。
├── random_img.py          # 工具腳本：從資料夾中隨機抽樣圖片，可用於建立測試集。
├── efficientnet_finetune_wbc.py # 實驗性腳本：探索使用 EfficientNet 進行遷移學習的訓練流程。
└── templates/
    └── index.html         # Web 應用的前端頁面。
環境設定
安裝 Python: 建議使用 Python 3.8 或更高版本。

安裝必要的套件:
您可以將以下內容存成 requirements.txt 檔案，然後執行 pip install -r requirements.txt 來安裝。

Plaintext

torch
torchvision
numpy
scikit-learn
opencv-python-headless
matplotlib
seaborn
tqdm
Flask
Flask-SocketIO
eventlet
Pillow
備註：若您使用 Mac M 系列晶片，PyTorch 會自動利用 MPS (Apple Silicon GPU) 進行加速。

使用教學
請依照以下步驟執行本專案：

步驟 1: 資料準備與前處理
準備資料:

將您所有的 .tiff 格式的白血球影像檔放在一個資料夾中（例如 img/data）。

請確保您的檔名格式為 標籤名_編號.tiff (例如: BAS_123.tiff, EOS_456.tiff)。

執行前處理腳本:
開啟終端機，執行 get_img.py。這個腳本會讀取您放在 TEST_DATA_PATH (請在腳本內修改) 的所有圖片，將它們轉換成統一大小，並將影像和標籤分別儲存成 processed_test_images.npy 和 processed_test_labels.npy。

Bash

python get_img.py
步驟 2: 模型訓練
設定路徑: 打開 cnn_model.py，確認 .npy 檔案的讀取路徑和模型儲存的路徑是正確的。

開始訓練:
執行 cnn_model.py 來開始訓練您的模型。訓練過程中，腳本會自動將資料集分為訓練集和驗證集，並在每個 epoch 結束後顯示當前的損失值（loss）和準確率（accuracy）。表現最好的模型將會被儲存下來。

Bash

python cnn_model.py
步驟 3: 模型評估
訓練完成後，您可以執行 test_model.py 來評估模型在測試資料上的效能。

設定路徑: 打開 test_model.py，確認載入的模型路徑 (MODEL_PATH) 和測試資料路徑是正確的。

執行評估:

Bash

python test_model.py
執行後，終端機會印出詳細的分類報告，其中包含了每個類別的精確率（Precision）、召回率（Recall）和 F1-score。同時，會彈出一個混淆矩陣的視覺化圖表，幫助您直觀地分析模型的預測情況。

步驟 4: 啟動即時分類網頁
您可以啟動一個本地伺服器，透過網頁來測試您的模型。

設定模型路徑: 打開 server.py，確保 MODEL_PATH 指向您訓練好的模型檔案。

啟動伺服器:

Bash

python server.py
伺服器啟動後，請在瀏覽器中打開 ”http://127.0.0.1:5001“
您會看到一個即時影像串流的畫面，模型會對畫面中的影像進行分類，並在下方顯示預測的白血球類別。

未來方向
模型優化: 持續調整目前模型的超參數，或可將實驗性的 Inception 模組或 EfficientNet 遷移學習流程進一步完善，以期達到更高的分類準確率。

擴充資料集: 增加更多樣化、更大量的訓練資料，是提升模型效能最直接有效的方法。

使用者介面優化: 強化前端介面，例如加入上傳圖片功能、顯示各類別的預測機率等，提供更豐富的使用者體驗。

