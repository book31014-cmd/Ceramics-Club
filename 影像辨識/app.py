import torch
from PIL import Image
import exifread
import glob
import os
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from datetime import datetime
import shutil

# --- 1. 設定 (針對雲端環境修改) ---
DB_DIR = "舊照片庫"
MODEL_NAME = "clip-ViT-B-32"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 載入 AI 大腦 ---
@st.cache_resource # 使用 Streamlit 快取避免重複載入模型
def load_model():
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
        return model
    except Exception as e:
        st.error(f"載入 AI 大腦失敗: {e}")
        st.stop()

model = load_model()

# --- 3. 準備舊照片記憶 (使用您融合後的函數) ---
def get_image_features(image_paths):
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            images.append(Image.open(path).convert("RGB"))
            valid_paths.append(path)
        except Exception as e:
            # 在 Streamlit 中使用 st.warning 顯示錯誤
            st.warning(f"無法開啟圖片 {os.path.basename(path)}: {e}")
            
    if not images:
        st.error(f"錯誤: 在 '{DB_DIR}' 資料夾中找不到可用的圖片。")
        st.stop()

    features = model.encode(images, convert_to_tensor=True, show_progress_bar=False)
    return features, valid_paths

def get_exif_time(image_path):
    # ... (使用您融合後的 get_exif_time 函數，程式碼同上一則訊息) ...
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            if 'EXIF DateTimeOriginal' in tags:
                return str(tags['EXIF DateTimeOriginal'])
            elif 'Image DateTime' in tags:
                 return str(tags)
            elif 'DateTime' in tags:
                 return str(tags['DateTime'])
    except Exception as e:
        pass
    try:
        m_time_timestamp = os.path.getmtime(image_path)
        return datetime.fromtimestamp(m_time_timestamp).strftime('%Y:%m:%d %H:%M:%S')
    except Exception as e:
        pass
    return "未知時間"

# 取得資料庫中的所有圖片路徑
db_image_paths = glob.glob(os.path.join(DB_DIR, '*.jpg')) + \
                 glob.glob(os.path.join(DB_DIR, '*.png'))

if not db_image_paths:
    st.error(f"錯誤: 在 '{DB_DIR}' 資料夾中找不到任何圖片。")
    st.stop()

db_features, db_valid_paths = get_image_features(db_image_paths)


# --- 4. 比對新照片 (網頁介面邏輯) ---

st.title("🖼️ AI 圖片相似度比對器")
st.write(f"資料庫中共有 **{len(db_valid_paths)}** 張圖片準備就緒。")

uploaded_file = st.file_uploader("請選擇一張新照片上傳進行比對...", type=["jpg", "png"])

if uploaded_file is not None:
    # 將上傳的檔案暫存起來供PIL開啟
    with open(os.path.join("./temp_upload", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    new_photo_path = os.path.join("./temp_upload", uploaded_file.name)

    try:
        # 處理新照片的特徵
        new_photo = Image.open(new_photo_path).convert("RGB")
        new_photo_feature = model.encode(new_photo, convert_to_tensor=True)

        # 計算相似度分數
        cos_scores = util.cos_sim(new_photo_feature, db_features)
        best_match_idx_scalar = torch.argmax(cos_scores).item()
        best_score = cos_scores.flatten()[best_match_idx_scalar].item()
        best_match_path = db_valid_paths[best_match_idx_scalar]
        best_match_time = get_exif_time(best_match_path)

        # 報告結果
        st.subheader("比對結果")
        st.image(new_photo, caption=f"您上傳的新照片: {uploaded_file.name}", width=200)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"資料庫中最相似的照片: **{os.path.basename(best_match_path)}**")
            st.write(f"相似度分數: **{best_score:.4f}**")
            st.write(f"那張舊照片的拍攝時間: **{best_match_time}**")
        with col2:
            st.image(Image.open(best_match_path).convert("RGB"), caption="資料庫中的匹配照片", width=200)

        st.markdown("---")
        if best_score > 0.85:
            st.success(f"🎉 結論: AI 認為這**很可能**是同一件作品！")
        else:
            st.info(f"🤔 結論: 相似度不高，可能是一件全新的作品喔！")

    except Exception as e:
        st.error(f"處理照片時發生錯誤: {e}")
    finally:
        # 清理暫存檔案
        if os.path.exists("./temp_upload"):
            shutil.rmtree("./temp_upload")

