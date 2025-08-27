# ===================================================================
# 最終正式版 get_img.py (含進度條)
# ===================================================================
import os
import cv2
import numpy as np
from tqdm import tqdm # 匯入 tqdm
import numpy as np
TEST_DATA_PATH = "/Users/wdwddaniel/Desktop/side_projects/ALL_cells/img/test"
SAVE_PATH = "/Users/wdwddaniel/Desktop/side_projects/ALL_cells/processed_data"
# --- 設定區 ---
IMAGE_SIZE = (128, 128)
CELL_TYPES = {
    '1': 'BAS', '2': 'EBO', '3': 'EOS', '4': 'KSC', '5': 'LYA',
    '6': 'LYT', '7': 'MMZ', '8': 'MOB', '9': 'MON', '10': 'MYB',
    '11':'MYO', '12':'NGB', '13':'NGS', '14':'PMB', '15':'PMO',
}
def load_data_from_filenames(data_path, image_size):
    """
    從檔名中讀取影像和標籤 (修正版)
    """
    images = []
    labels = []
    
    # 建立一個有效標籤名稱的集合，方便快速查找
    VALID_LABELS = set(CELL_TYPES.values())

    print(f"正在載入 {data_path} 裡的影像並解析檔名標籤...")
    
    file_list = os.listdir(data_path)
    for filename in tqdm(file_list, desc="處理影像中"):
        if not filename.lower().endswith(".tiff"):
            continue

        # --- 1. 先解析標籤，並檢查格式是否正確 ---
        try:
            label_name = filename.split('_')[0]
        except IndexError:
            # 如果檔名中沒有 '_', 跳過此檔案
            continue

        # --- 2. 檢查解析出的標籤是否為我們定義的有效細胞類型 ---
        if label_name not in VALID_LABELS:
            continue
        
        # --- 3. 讀取影像，並檢查是否成功 ---
        image_path = os.path.join(data_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        # --- 4. 所有檢查都通過後，才進行處理並「成對地」加入列表 ---
        resized_image = cv2.resize(image, image_size)
        images.append(resized_image)
        labels.append(label_name) # 我們加入的是細胞名稱，例如 'EOS'
    
    print(f"處理完成！共成功載入 {len(images)} 張影像和 {len(labels)} 個標籤。")
    assert len(images) == len(labels) # 最終防呆確認

    # --- 後續處理 ---
    images_np = np.array(images, dtype=np.float32) / 255.0
    
    # 建立標籤名稱到數字的映射
    unique_labels_list = sorted(list(np.unique(labels)))
    label_to_int = {label: i for i, label in enumerate(unique_labels_list)}
    
    # 將文字標籤列表轉換為數字標籤列表
    labels_int = np.array([label_to_int[label] for label in labels])

    return images_np, labels_int, unique_labels_list

# ----------------- 新增的程式碼區塊 -----------------

# 設定測試集路徑


# 載入並處理測試集資料
test_images, test_labels, test_unique_labels = load_data_from_filenames(TEST_DATA_PATH, (128, 128))

# 定義儲存測試集的檔案路徑
PROCESSED_TEST_IMAGES_FILE = os.path.join(SAVE_PATH, "processed_test_images.npy")
PROCESSED_TEST_LABELS_FILE = os.path.join(SAVE_PATH, "processed_test_labels.npy")

print("---")
print("開始將測試資料儲存為 .npy 檔案...")

# 儲存測試集資料
np.save(PROCESSED_TEST_IMAGES_FILE, test_images)
np.save(PROCESSED_TEST_LABELS_FILE, test_labels)

print(f"測試集影像資料已儲存至: {PROCESSED_TEST_IMAGES_FILE}")
print(f"測試集標籤資料已儲存至: {PROCESSED_TEST_LABELS_FILE}")
print("儲存完成！")
