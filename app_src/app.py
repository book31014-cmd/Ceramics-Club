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
# 基本設定（免費版安全）
# ==============================
DB_DIR = "app_src/Photos"
DEVICE = "cpu"
MAX_DB_IMAGES = 9   # 🔴 免費版關鍵限制（一定要有）

st.set_page_config(
    page_title="AI 圖片相似度比對",
    layout="centered"
)

# ==============================
# 🎨 極簡 UI
# ==============================
st.markdown("""
<style>
.stApp { background: #f5f7fb; }

.card {
    background: white;
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 1.2rem;
}

h1 { font-size: 1.8rem; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ==============================
# 首頁
# ==============================
st.markdown("""
<div class="card">
<h1>🖼️ AI 圖片相似度比對系統</h1>
<p>使用 AI 分析圖片特徵，快速找到最相似的作品。</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# 載入 CLIP 模型（只一次）
# ==============================
@st.cache_resource
def load_clip():
    with st.spinner("🤖 載入 AI 模型中（首次稍久）..."):
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
# 載入舊照片特徵（限制數量）
# ==============================
@st.cache_data
def load_database():
    if not os.path.exists(DB_DIR):
        st.error(f"找不到資料夾：{DB_DIR}")
        st.stop()

    paths = (
        glob.glob(os.path.join(DB_DIR, "*.jpg")) +
        glob.glob(os.path.join(DB_DIR, "*.jpeg")) +
        glob.glob(os.path.join(DB_DIR, "*.png"))
    )

    # 🔴 免費版保命線
    paths = paths[:MAX_DB_IMAGES]

    if not paths:
        st.error("Photos 資料夾沒有圖片")
        st.stop()

    features = []
    valid_paths = []

    for p in paths:
        try:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                feat = model.encode_image(img.to(DEVICE))
                feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat)
            valid_paths.append(p)
        except:
            pass

    return torch.cat(features), valid_paths

db_features, db_paths = load_database()

st.markdown(f"""
<div class="card">
✅ 已載入 <b>{len(db_paths)}</b> 張舊照片（展示模式）
</div>
""", unsafe_allow_html=True)

# ==============================
# 📤 上傳新照片（唯一上傳框）
# ==============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📤 上傳照片進行比對")

uploaded = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

if uploaded:
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        st.image(uploaded, caption="你上傳的照片", width=320)

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
        st.subheader("🔍 比對結果")

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(best_path, caption="最相似的舊照片", use_container_width=True)

        with col2:
            st.metric("相似度", f"{best_score:.2f}")
            st.progress(int(best_score * 100))
            st.write(f"📄 檔名：{os.path.basename(best_path)}")
            st.write(f"📅 時間：{best_time}")

        st.markdown('</div>', unsafe_allow_html=True)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ==============================
# 🔐 Sidebar 管理者（免費版友善）
# ==============================
with st.sidebar:
    st.title("🔐 管理者")

    admin_upload = st.file_uploader(
        "新增舊照片（最多 10 張）",
        type=["jpg", "jpeg", "png"]
    )

    if admin_upload:
        if len(db_paths) >= MAX_DB_IMAGES:
            st.warning("已達展示上限（10 張）")
        else:
            save_path = os.path.join(DB_DIR, admin_upload.name)
            if os.path.exists(save_path):
                st.warning("檔名已存在")
            else:
                with open(save_path, "wb") as f:
                    f.write(admin_upload.getbuffer())

                st.success("已加入舊照片庫")
                st.cache_data.clear()
                st.rerun()


