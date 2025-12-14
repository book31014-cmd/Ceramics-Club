import torch
from PIL import Image
import exifread
import glob
import os
from sentence_transformers import SentenceTransformer, util
import sys
from datetime import datetime # 新增引入 datetime 模組

# --- 1. 設定 ---
DB_DIR = "舊照片庫" # 告訴程式我們的舊照片庫在哪裡
# 我們讓程式準備好接收一個新照片的名稱
if len(sys.argv) < 2:
    print("使用方式: python main.py [新照片路徑]")
    sys.exit(1)
NEW_PHOTO_PATH = sys.argv[1] # 取得您輸入的新照片名稱

MODEL_NAME = "clip-ViT-B-32" # 使用 AI 模型名稱
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 載入 AI 大腦 ---
try:
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"AI 助手已就緒，使用裝置：{device}")
except Exception as e:
    print(f"載入 AI 大腦失敗，請檢查網路連線或安裝: {e}")
    sys.exit(1)

# --- 3. 準備舊照片記憶 ---
def get_image_features(image_paths):
    # 使用 try-except 確保圖片能正常打開
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            # 轉換為 RGB 確保與模型相容
            images.append(Image.open(path).convert("RGB"))
            valid_paths.append(path)
        except Exception as e:
            print(f"無法開啟圖片 {path}: {e}")
            
    if not images:
        print("舊照片庫中沒有可用的圖片。")
        sys.exit(1)

    # 將圖片轉換為 AI 能理解的特徵向量
    features = model.encode(images, convert_to_tensor=True, show_progress_bar=False)
    return features, valid_paths

# 融合後的 get_exif_time 函數 (包含 EXIF 讀取及檔案修改時間備援)
def get_exif_time(image_path):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            # 嘗試獲取多個可能的日期標籤
            if 'EXIF DateTimeOriginal' in tags:
                return str(tags['EXIF DateTimeOriginal'])
            elif 'Image DateTime' in tags:
                 return str(tags)
            elif 'DateTime' in tags:
                 return str(tags['DateTime'])
    except Exception as e:
        pass
    
    # 如果找不到 EXIF 資訊，嘗試讀取檔案的修改時間 (Fallback Option)
    try:
        m_time_timestamp = os.path.getmtime(image_path)
        # 格式化輸出以符合 EXIF 的標準時間格式
        return datetime.fromtimestamp(m_time_timestamp).strftime('%Y:%m:%d %H:%M:%S')
    except Exception as e:
        pass

    return "未知時間"


# 取得資料庫中的所有圖片路徑
db_image_paths = glob.glob(os.path.join(DB_DIR, '*.jpg')) + \
                 glob.glob(os.path.join(DB_DIR, '*.png'))

if not db_image_paths:
    print(f"錯誤: 在 '{DB_DIR}' 資料夾中找不到任何圖片。")
    sys.exit(1)

db_features, db_valid_paths = get_image_features(db_image_paths)


# --- 4. 比對新照片 ---
def find_similar_photo(new_photo_path):
    try:
        # 處理新照片的特徵
        new_photo = Image.open(new_photo_path).convert("RGB")
        new_photo_feature = model.encode(new_photo, convert_to_tensor=True)
    except Exception as e:
        print(f"錯誤: 無法處理新照片 '{new_photo_path}': {e}")
        sys.exit(1)


    # 計算相似度分數 (餘弦相似度，分數越高越像)
    cos_scores = util.cos_sim(new_photo_feature, db_features)
    
    # --- 修正此處的邏輯，避免 RuntimeError: a Tensor with 2 elements cannot be converted to Scalar ---
    # 獲取最高分數的索引（將索引轉換為純量整數）
    best_match_idx_scalar = torch.argmax(cos_scores).item()
    
    # 從分數矩陣中取出該單一最高分數
    # cos_scores.flatten() 將 1xN 轉為 N 個元素的向量
    best_score = cos_scores.flatten()[best_match_idx_scalar].item()
    best_match_path = db_valid_paths[best_match_idx_scalar]
    # ------------------------------------------------------------------------------------

    best_match_time = get_exif_time(best_match_path)

    # 報告結果
    print("-" * 40)
    print(f"您要求比對的新照片: {os.path.basename(new_photo_path)}")
    print(f"資料庫中最相似的照片: {os.path.basename(best_match_path)}")
    print(f"相似度分數 (滿分 1.0): {best_score:.4f}")
    print(f"那張舊照片的拍攝時間: {best_match_time}")
    print("-" * 40)
    
    if best_score > 0.85: # 如果相似度很高 (超過 85%)
        print(f"🎉 結論: AI 認為這**很可能**是同一件作品！您上次拍它是在 {best_match_time}。")
    else:
        print(f"🤔 結論: 相似度不高，可能是一件全新的作品喔！")


# --- 5. 啟動比對 ---
find_similar_photo(NEW_PHOTO_PATH)
