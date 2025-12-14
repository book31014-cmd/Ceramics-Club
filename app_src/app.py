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

st.set_page_config(
    page_title="AI 圖片比對助手",
    layout="centered"
)

# ==============================
# 🎨 全站 UI 美化（CSS）
# ==============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8f9fa, #eef2f7);
}

.card {
    background: white;
    padding: 1.6rem;
    border-radius: 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.08);
    margin-bottom: 1.6rem;
}

.admin {
    border: 2px dashed #cbd5e1;
    background: #fafafa;
}

h1 {
    font-weight: 800;
    letter-spacing: 1px;
}

.badge {
    display: inline-block;
    padding: 0.3em 0.8em;
    border-radius: 999px;
    background: #4CAF50;
    color: white;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🏠 首頁介紹
# ==============================
st.markdown("""
<div class="card">
<h1>🖼️ AI 圖片相似度比對系統</h1>
<p>
本系統結合 <b>OpenCLIP AI 視覺模型</b>，<br>
可用於 <b>陶藝作品管理、相似作品搜尋與比對</b>。
</p>
<span class="badge">AI Image Retrieval</span>
</div>
""", unsafe_allow_html=True)

# ==============================
# 載入 CLIP 模型
# ==============================
@st.cache_resource
def load_clip():
    with st.spinner("🤖 載入 AI 圖片模型中（首次較久）..."):
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
# 載入舊照片特徵庫
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

    progress = st.progress(0.0, "📂 建立圖片特徵庫中...")
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

st.markdown(f"""
<div class="card">
✅ 已載入 <b>{len(db_paths)}</b> 張舊照片
</div>
""", unsafe_allow_html=True)

# ==============================
# 📤 上傳新照片比對
# ==============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📤 上傳新照片進行比對")

uploaded = st.file_uploader(
    "支援 JPG / PNG，請選擇一張照片",
    type=["jpg", "jpeg", "png"]
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded:
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        st.image(uploaded, caption="您上傳的照片", width=320)

        with st.spinner("🔍 AI 比對中..."):
            img = preprocess(Image.open(temp_path).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                q_feat = model.encode_image(img.to(DEVICE))
                q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)

            scores = (q_feat @ db_features.T).squeeze(0)
            idx = torch.argmax(scores).item()

            best_path = db_paths[idx]
            best_score = scores[idx].item()
            best_time = get_exif_time(best_path)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 AI 比對結果")

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(best_path, caption="最相似的舊照片", use_container_width=True)

        with col2:
            st.metric("相似度", f"{best_score:.2f}")
            st.progress(int(best_score * 100))
            st.write(f"📄 **檔名**：{os.path.basename(best_path)}")
            st.write(f"📅 **時間**：{best_time}")

        st.markdown('</div>', unsafe_allow_html=True)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ==============================
# 🔐 管理者功能
# ==============================
st.markdown('<div class="card admin">', unsafe_allow_html=True)
st.subheader("🔐 管理者功能｜新增舊照片")
st.caption("此功能用於展示與管理，重新部署後需重新上傳")

admin_upload = st.file_uploader(
    "選擇要加入舊照片庫的圖片",
    type=["jpg", "jpeg", "png"],
    key="admin_uploader"
)

if admin_upload:
    save_path = os.path.join(DB_DIR, admin_upload.name)

    if os.path.exists(save_path):
        st.warning("⚠️ 檔名已存在，請更換後再上傳")
    else:
        with open(save_path, "wb") as f:
            f.write(admin_upload.getbuffer())

        st.success(f"✅ 已加入舊照片庫：{admin_upload.name}")
        st.cache_data.clear()
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)





