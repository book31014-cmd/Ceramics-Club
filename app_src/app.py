import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch
import os
import glob
import exifread
from datetime import datetime
import shutil

# --- 1. 設定 (針對雲端環境修改) ---
# 注意：請確保 GitHub 上您的 app.py 同層目錄下真的有一個叫做 "舊照片庫" 的資料夾
DB_DIR = 'photos'
MODEL_NAME = "clip-ViT-B-32"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 設定頁面標題
st.set_page_config(page_title="AI 圖片比對助手", layout="centered")

# --- 2. 載入 AI 大腦 ---
@st.cache_resource # 使用 Streamlit 快取避免重複載入模型
def load_model():
    try:
        # 顯示載入中的狀態
        with st.spinner('正在喚醒 AI 大腦... (第一次啟動需要一點時間)'):
            model = SentenceTransformer(MODEL_NAME, device=device)
        return model
    except Exception as e:
        st.error(f"載入 AI 大腦失敗: {e}")
        st.stop()

model = load_model()

# --- 3. 準備舊照片記憶 ---
def get_image_features(image_paths):
    images = []
    valid_paths = []
    
    # 建立進度條，因為處理圖片可能需要時間
    progress_bar = st.progress(0, text="正在讀取資料庫圖片...")
    
    for i, path in enumerate(image_paths):
        try:
            images.append(Image.open(path).convert("RGB"))
            valid_paths.append(path)
        except Exception as e:
            st.warning(f"無法開啟圖片 {os.path.basename(path)}: {e}")
        
        # 更新進度條
        progress_bar.progress((i + 1) / len(image_paths))
            
    progress_bar.empty() # 讀取完成後隱藏進度條

    if not images:
        st.error(f"錯誤: 在 '{DB_DIR}' 資料夾中找不到可用的圖片。請檢查 GitHub 資料夾結構。")
        st.stop()

    features = model.encode(images, convert_to_tensor=True, show_progress_bar=False)
    return features, valid_paths

def get_exif_time(image_path):
    # 嘗試讀取 EXIF 資訊
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            if 'EXIF DateTimeOriginal' in tags:
                return str(tags['EXIF DateTimeOriginal'])
            elif 'Image DateTime' in tags:
                 return str(tags['Image DateTime'])
            elif 'DateTime' in tags:
                 return str(tags['DateTime'])
    except Exception:
        pass
    
    # 如果沒有 EXIF，嘗試讀取檔案修改時間
    try:
        m_time_timestamp = os.path.getmtime(image_path)
        return datetime.fromtimestamp(m_time_timestamp).strftime('%Y:%m:%d %H:%M:%S')
    except Exception:
        pass
        
    return "未知時間"

# --- 主程式邏輯 ---

st.title("🖼️ AI 圖片相似度比對器")

# 檢查資料庫資料夾是否存在
if not os.path.exists(DB_DIR):
    st.error(f"嚴重錯誤：找不到資料夾 '{DB_DIR}'。請確認 GitHub 上資料夾名稱是否完全一致。")
    st.stop()

# 取得資料庫中的所有圖片路徑
db_image_paths = glob.glob(os.path.join(DB_DIR, '*.jpg')) + \
                 glob.glob(os.path.join(DB_DIR, '*.png')) + \
                 glob.glob(os.path.join(DB_DIR, '*.jpeg'))

if not db_image_paths:
    st.error(f"錯誤: 在 '{DB_DIR}' 資料夾中找不到任何圖片。")
    st.stop()

# 計算資料庫特徵 (只會執行一次)
db_features, db_valid_paths = get_image_features(db_image_paths)

st.write(f"✅ 資料庫載入完成！共有 **{len(db_valid_paths)}** 張舊照片準備比對。")
st.markdown("---")

# --- 4. 比對新照片 (網頁介面邏輯) ---

uploaded_file = st.file_uploader("👉 請選擇一張新照片上傳進行比對...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    temp_dir = "./temp_upload"
    
    # --- 關鍵修正開始: 確保暫存資料夾存在 ---
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # --- 關鍵修正結束 ---

    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    # 將上傳的檔案暫存起來供 PIL 開啟
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # 顯示使用者上傳的圖片
        st.image(uploaded_file, caption=f"您上傳的新照片", width=300)
        
        with st.spinner('AI 正在進行特徵比對...'):
            # 處理新照片的特徵
            new_photo = Image.open(temp_file_path).convert("RGB")
            new_photo_feature = model.encode(new_photo, convert_to_tensor=True)

            # 計算相似度分數
            cos_scores = util.cos_sim(new_photo_feature, db_features)
            
            # 找出最高分的
            best_match_idx = torch.argmax(cos_scores).item()
            best_score = cos_scores[0][best_match_idx].item()
            best_match_path = db_valid_paths[best_match_idx]
            best_match_time = get_exif_time(best_match_path)

        # 報告結果
        st.divider()
        st.subheader("🔍 比對結果")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.info("🏆 最相似的舊照片")
            st.image(Image.open(best_match_path).convert("RGB"), caption="資料庫中的匹配照片", use_container_width=True)
            
        with col2:
            st.write(f"📄 檔案名稱: **{os.path.basename(best_match_path)}**")
            st.write(f"📅 拍攝/建立時間: **{best_match_time}**")
            
            # 顯示相似度分數條
            st.write("📊 相似度分數:")
            st.progress(int(best_score * 100))
            st.write(f"**{best_score:.4f}** (滿分 1.0)")

        st.markdown("---")
        
        # 根據分數給出結論
        if best_score > 0.85:
            st.success(f"🎉 **高度相似！** AI 認為這極有可能是同一件作品或同一場景。")
        elif best_score > 0.7:
            st.warning(f"🤔 **中度相似。** 可能是類似的風格或構圖，但不一定是同一張。")
        else:
            st.info(f"🆕 **相似度低。** 這看起來是一件全新的作品喔！")

    except Exception as e:
        st.error(f"處理照片時發生錯誤: {e}")
        
    finally:
        # 清理暫存檔案 (保持環境整潔)
        if os.path.exists(temp_dir):

            shutil.rmtree(temp_dir)


