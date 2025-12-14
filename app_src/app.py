import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch
import os
import glob
import exifread
from datetime import datetime
import shutil

# ==============================
# 1. 基本設定（雲端安全版）
# ==============================
DB_DIR = "app_src/Photos"
MODEL_NAME = "clip-ViT-B-32-multilingual-v1"
DEVICE = "cpu"  # 🚨 Streamlit Cloud 一律用 CPU

st.set_page_config(page_title="AI 圖片比對助手", layout="centered")
st.title("🖼️ AI 圖片相似度比對器")

# ==============================
# 2. 載入模型（快取，避免崩潰）
# ==============================
@st.cache_resource
def load_model():
    with st.spinner("正在載入 AI 模型（首次啟動較久）..."):
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return model

model = load_model()

# ==============================
# 3. EXIF 讀取
# ==============================
def get_exif_time(image_path):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            for key in ['EXIF DateTimeOriginal', 'Image DateTime', 'DateTime']:
                if key in tags:
                    return str(tags[key])
    except Exception:
        pass

    try:
        return datetime.fromtimestamp(os.path.getmtime(image_path)).strftime('%Y:%m:%d %H:%M:%S')
    except Exception:
        return "未知時間"

# ==============================
# 4. 載入資料庫圖片特徵（只算一次）
# ==============================
@st.cache_data
def load_database():
    if not os.path.exists(DB_DIR):
        st.error(f"❌ 找不到資料夾：{DB_DIR}")
        st.stop()

    image_paths = (
        glob.glob(os.path.join(DB_DIR, "*.jpg")) +
        glob.glob(os.path.join(DB_DIR, "*.png")) +
        glob.glob(os.path.join(DB_DIR, "*.jpeg"))
    )

    if not image_paths:
        st.error("❌ Photos 資料夾內沒有圖片")
        st.stop()

    images = []
    valid_paths = []

    progress = st.progress(0.0, "讀取資料庫圖片中...")
    for i, p in enumerate(image_paths):
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_paths.append(p)
        except:
            pass
        progress.progress((i + 1) / len(image_paths))
    progress.empty()

    features = model.encode(images, convert_to_tensor=True)
    return features, valid_paths

db_features, db_paths = load_database()
st.success(f"✅ 已載入 {len(db_paths)} 張舊照片")
st.divider()

# ==============================
# 5. 上傳 & 比對
# ==============================
uploaded_file = st.file_uploader(
    "👉 上傳一張新照片進行比對",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        st.image(uploaded_file, caption="您上傳的照片", width=300)

        with st.spinner("AI 正在比對中..."):
            query_img = Image.open(temp_path).convert("RGB")
            query_feat = model.encode(query_img, convert_to_tensor=True)

            scores = util.cos_sim(query_feat, db_features)
            idx = torch.argmax(scores).item()

            best_path = db_paths[idx]
            best_score = scores[0][idx].item()
            best_time = get_exif_time(best_path)

        st.subheader("🔍 比對結果")
        col1, col2 = st.columns(2)

        with col1:
            st.image(best_path, caption="最相似的舊照片", use_container_width=True)

        with col2:
            st.write(f"📄 檔名：**{os.path.basename(best_path)}**")
            st.write(f"📅 時間：**{best_time}**")
            st.write("📊 相似度")
            st.progress(int(best_score * 100))
            st.write(f"**{best_score:.4f}**")

        if best_score > 0.85:
            st.success("🎉 高度相似")
        elif best_score > 0.7:
            st.warning("🤔 中度相似")
        else:
            st.info("🆕 相似度低")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


