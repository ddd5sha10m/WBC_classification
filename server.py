from flask import Flask, render_template
from flask_socketio import SocketIO
import torch
import cv2
import numpy as np
import base64
import eventlet # 確保 eventlet 被匯入

# <-- 從 model_definition.py 檔案中匯入您定義好的模型架構
from model_definition import CellCNN

# --- 1. 初始化 Flask 和 SocketIO ---
app = Flask(__name__, template_folder='./templates', static_folder='./static')
socketio = SocketIO(app)

# --- 2. 設備設定 ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("後端伺服器將使用 MPS (GPU) 進行運算。")
else:
    device = torch.device("cpu")
    print("後端伺服器將使用 CPU 進行運算。")

# --- 3. 載入訓練好的模型 ---
NUM_CLASSES = 15 
MODEL_PATH = "/Users/wdwddaniel/Desktop/side_projects/ALL_cells/model/cell_cnn_model2.pth"
# 使用匯入的 CellCNN 類別來實例化模型
model = CellCNN(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # 設定為評估模式
print("PyTorch 模型載入成功！")

# --- 4. 定義細胞類別名稱 ---
CELL_TYPES = ['BAS', 'EBO', 'EOS', 'KSC', 'LYA', 'LYT', 'MMZ', 'MOB', 'MON', 'MYB', 'MYO', 'NGB', 'NGS', 'PMB', 'PMO']

# --- 5. 設定路由與 WebSocket 事件 ---
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(data_image):
    # 解碼 Base64 影像
    sbuf = data_image.split(',')[1]
    image_data = base64.b64decode(sbuf)
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 影像預處理
    resized_frame = cv2.resize(frame, (128, 128))
    tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0).to(device)
    
    # 執行模型預測
    with torch.no_grad():
        output = model(tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_label = CELL_TYPES[predicted_idx.item()]
    
    # 將結果傳回前端
    socketio.emit('response', {'prediction': predicted_label})

# --- 6. 啟動伺服器 ---
if __name__ == '__main__':
    print("伺服器啟動中... 請在瀏覽器中打開 http://127.0.0.1:5001")
    # 使用 eventlet 來啟動，以獲得最佳的 WebSocket 效能
    socketio.run(app, host='0.0.0.0', port=5001)