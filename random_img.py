import os
import random
import shutil

# 設定來源資料夾與目標資料夾
source_folder = "/Users/wdwddaniel/Desktop/side_projects/ALL_cells/img/data"   # 舊的資料夾
target_folder = "/Users/wdwddaniel/Desktop/side_projects/ALL_cells/img/test"   # 新的資料夾

# 如果新資料夾不存在，就建立它
os.makedirs(target_folder, exist_ok=True)

# 取得來源資料夾裡所有檔案
all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# 只取圖片檔 (可依需要加副檔名過濾)
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
image_files = [f for f in all_files if f.lower().endswith(image_extensions)]

# 隨機挑選 2000 個 (如果檔案數小於2000，就全取)
sample_size = min(2000, len(image_files))
selected_files = random.sample(image_files, sample_size)

# 複製檔案到新資料夾
for file_name in selected_files:
    src_path = os.path.join(source_folder, file_name)
    dst_path = os.path.join(target_folder, file_name)
    shutil.copy(src_path, dst_path)

print(f"已隨機挑選 {sample_size} 個圖片，並複製到 {target_folder}")
