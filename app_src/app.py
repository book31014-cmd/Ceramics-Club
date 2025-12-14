# ==============================
# 6. 🔐 新增舊照片到資料庫（管理功能）
# ==============================
st.divider()
st.subheader("📥 新增舊照片到資料庫（管理功能）")

st.caption("⚠️ 此功能用於展示與管理，重新部署後需重新上傳")

admin_upload = st.file_uploader(
    "選擇要加入舊照片庫的圖片（JPG / PNG）",
    type=["jpg", "jpeg", "png"],
    key="admin_uploader"
)

if admin_upload:
    save_path = os.path.join(DB_DIR, admin_upload.name)

    if os.path.exists(save_path):
        st.warning("⚠️ 此檔名已存在，請更換檔名後再上傳")
    else:
        with open(save_path, "wb") as f:
            f.write(admin_upload.getbuffer())

        st.success(f"✅ 已加入舊照片庫：{admin_upload.name}")
        st.info("🔄 正在重新載入資料庫，請稍候...")

        # 清除快取，強制重新計算特徵
        st.cache_data.clear()

      


