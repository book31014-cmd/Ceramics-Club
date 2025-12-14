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
# 1) 基本設定（免費版穩定）
# ==============================
DB_DIR = "app_src/Photos"
DEVICE = "cpu"
MAX_DB_IMAGES = 10  # ✅ 免費版保命上限：最多載入 10 張舊照片

st.set_page_config(page_title="AI 圖片相似度比對", layout="centered")

# ==============================
# 2) UI（簡單乾淨、穩）
# ==============================
st.markdown("""
<style>
.stApp { background: #f5f7fb; }
.card {
  background: #fff;
  padding: 1.3rem 1.5rem;
  border-radius: 14px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
  margin-bottom: 1.0rem;
}
h1 { font-size: 1.8rem; margin: 0 0 .25rem 0; }
.small { color: #6b7280; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h1>🖼️ AI 圖片相似度比對器</h1>
<div class="small">上傳一張新照片，AI 會找出舊照片庫中最相似的一張。</div>
</div>
""", unsafe_allow_html=True)

# ==============================
# 3) 載入 OpenCLIP（只一次）
# ==============================
@st.cache_resource
def load_clip():
    with st.spinner("🤖 載入 AI 模型中（首次較久）..."):
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai"
        )
        model = model.to(DEVICE)
        model.eval()
    return model, preprocess

model, preprocess = load_clip()

# ==============================
# 4) EXIF 時間
# ==============================
def get_exif_time(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f)
            for key in ["EXIF DateTimeOriginal", "Image DateTime", "DateTime"]:
                if key in tags:
                    return str(tags[key])
    except Exception:
        pass

    try:
        return datetime.fromtimestamp(os.path.getmtime(image_path)).strftime("%Y:%m:%d %H:%M:%S")
    except Exception:
        return "未知時間"

# ==============================
# 5) 載入舊照片特徵庫（限制數量，穩定）
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

    # ✅ 免費版保命：限制最多 N 張
    paths = paths[:MAX_DB_IMAGES]

    features = []
    valid_paths = []

    # ✅ 不用 progress（也省資源），要更穩
    for p in paths:
        try:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                feat = model.encode_image(img.to(DEVICE))
                feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat)
            valid_paths.append(p)
        except Exception:
            pass

    if not valid_paths:
        st.error("❌ 圖片讀取失敗（可能格式損壞）")
        st.stop()

    return torch.cat(features), valid_paths

db_features, db_paths = load_database()

st.markdown(f"""
<div class="card">
✅ 已載入 <b>{len(db_paths)}</b> 張舊照片（展示模式：最多 {MAX_DB_IMAGES} 張）
</div>
""", unsafe_allow_html=True)

# ==============================
# 5-2) 上傳新照片比對（主功能）
# ==============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📤 上傳新照片進行比對")

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
# 6) 🔐 新增舊照片到資料庫（管理功能）— 穩定版
# ==============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔐 管理功能：新增舊照片")

st.caption("⚠️ 展示用功能：重新部署後需重新上傳。免費版限制：舊照片最多 10 張。")

admin_upload = st.file_uploader(
    "選擇要加入舊照片庫的圖片（JPG / PNG）",
    type=["jpg", "jpeg", "png"],
    key="admin_uploader"
)

if admin_upload:
    # ✅ 達上限就不讓加，避免爆資源
    if len(db_paths) >= MAX_DB_IMAGES:
        st.warning(f"已達展示上限（{MAX_DB_IMAGES} 張）。請先移除一些圖片或提高上限（可能會爆資源）。")
    else:
        save_path = os.path.join(DB_DIR, admin_upload.name)

        if os.path.exists(save_path):
            st.warning("⚠️ 此檔名已存在，請更換檔名後再上傳")
        else:
            with open(save_path, "wb") as f:
                f.write(admin_upload.getbuffer())

            st.success(f"✅ 已加入舊照片庫：{admin_upload.name}")
            st.info("🔄 重新建立特徵庫中...")

            # ✅ 清快取，讓 load_database 重新跑（但因為有上限，所以穩）
            st.cache_data.clear()
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)




