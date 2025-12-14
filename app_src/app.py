import streamlit as st
import torch
import open_clip
from PIL import Image
import os
import glob
import exifread
from datetime import datetime
import shutil

# ==============================
# 基本設定
# ==============================
DB_DIR = "app_src/Photos"
DEVICE = "cpu"

st.set_page_config(page_title="AI 圖片比對助手", layout="centered")
st.title("🖼️ AI 圖片相似度比對器")

# ==============================
# 載入 CLIP（真正的圖片模型）
# ==============================
@st.cache_resource
def load_clip():
    with st.spinner("載入 AI 圖片模型中（首次較久）..."):
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai"
        )
        model = model.to(DEVICE)
        model.eval()
    return model, preprocess

model, preprocess = load_clip()

# ==============================
# EXIF 時間
# ==============================
def get_exif_time(image_path):
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f)
            for key in ["EXIF DateTimeOriginal", "Image DateTime", "DateTime"]:
                if key in tags:
                    return str(tags[key])
    except:
        pass

    try:
        return datetime.fromtimestamp(
            os.path.getmtime(image_path)
        ).strftime("%Y:%m:%d %H:%M:%S")
    except:
        return "未知時間"

# ==============================
# 載入資料庫圖片特徵
# ==============================
@st.cache_data
def load_database():
    if not os.path.exists(DB_DIR):
        st.error(f"❌ 找不到資料夾：{DB_DIR}")
        st.stop()

    paths = (
        glob.glob(os.path.join(DB_DIR, "*.jpg")) +
        glob.glob(os.path.join(DB_DIR, "*.jpeg")) +
        glob.glob(os.path.join(DB_DIR, "*.png"))
    )

    if not paths:
        st.error("❌ Photos 資料夾沒有圖片")
        st.stop()

    features = []
    valid_paths = []

    progress = st.progress(0.0, "正在建立圖片特徵庫...")
    for i, p in enumerate(paths):
        try:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                feat = model.encode_image(img.to(DEVICE))
                feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat)
            valid_paths.append(p)
        except:
            pass
        progress.progress((i + 1) / len(paths))
    progress.empty()

    return torch.cat(features), valid_paths

db_features, db_paths = load_database()
st.success(f"✅ 已載入 {len(db_paths)} 張舊照片")
st.divider()

# ==============================
# 上傳 & 比對
# ==============================
uploaded = st.file_uploader(
    "👉 上傳新照片進行比對",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        st.image(uploaded, caption="您上傳的照片", width=300)

        with st.spinner("AI 比對中..."):
            img = preprocess(Image.open(temp_path).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                q_feat = model.encode_image(img.to(DEVICE))
                q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)

            scores = (q_feat @ db_features.T).squeeze(0)
            idx = torch.argmax(scores).item()

            best_path = db_paths[idx]
            best_score = scores[idx].item()
            best_time = get_exif_time(best_path)

        st.subheader("🔍 比對結果")
        col1, col2 = st.columns(2)

        with col1:
            st.image(best_path, caption="最相似舊照片", use_container_width=True)

        with col2:
            st.write(f"📄 檔名：**{os.path.basename(best_path)}**")
            st.write(f"📅 時間：**{best_time}**")
            st.write("📊 相似度")
            st.progress(int(best_score * 100))
            st.write(f"**{best_score:.4f}**")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)




