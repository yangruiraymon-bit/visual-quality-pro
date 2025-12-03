import streamlit as st
from PIL import Image
import numpy as np
import cv2
from figure_ground_engine_v2 import FigureGroundEngineV2

# 假设上面的类在 figure_ground_engine_v2.py

st.set_page_config(page_title="全能图底关系分析", layout="wide", page_icon="🕵️")

st.title("🕵️ 全能图底关系分析 (Figure-Ground Pro)")
st.markdown("综合评估 **图形主体** 的突显程度与 **文字信息** 的易读性。")

uploaded_file = st.file_uploader("上传图片 (建议包含主体和文字)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    try:
        image_pil = Image.open(uploaded_file)
        img_np = np.array(image_pil.convert('RGB'))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        with st.spinner("正在进行双重分析 (主体分割 + 文字OCR)..."):
            if 'fg_engine' not in st.session_state:
                st.session_state.fg_engine = FigureGroundEngineV2()
            report = st.session_state.fg_engine.analyze(img_bgr)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("分析视图")
            st.image(cv2.cvtColor(report.visualization, cv2.COLOR_BGR2RGB), 
                     caption="绿框=文字易读 | 红框=文字难辨", use_container_width=True)
        with col2:
            st.subheader("📊 综合评分")
            st.metric("全局图底质量", f"{report.overall_score} / 100")
            st.divider()
            st.markdown("#### 🖼️ 图形主体 (Macro)")
            gen = report.general
            c1, c2 = st.columns(2)
            c1.metric("色彩分离度", f"{gen.color_diff}", delta=">60 优" if gen.is_strong else "弱")
            c2.metric("面积主导性", f"{gen.area_diff}", help="前景vs背景面积差")
            if gen.is_strong:
                st.success("✅ 图形主体非常突出，视觉焦点明确。")
            else:
                st.warning("⚠️ 图形主体与背景融合，视觉焦点不强。")
            st.divider()
            st.markdown("#### 🔤 文字信息 (Micro)")
            if report.text_regions:
                legible_count = sum(1 for t in report.text_regions if t.is_legible)
                total_count = len(report.text_regions)
                st.metric("易读文字比例", f"{legible_count}/{total_count}", 
                          delta="需优化" if legible_count < total_count else "完美")
                with st.expander("查看详细文字数据"):
                    for t in report.text_regions:
                        icon = "✅" if t.is_legible else "🔴"
                        st.write(f"**{icon} '{t.text}'**")
                        st.caption(f"对比度: {t.local_contrast} | 背景噪点: {t.bg_noise}")
                        if not t.is_legible:
                            if t.local_contrast < 70:
                                st.write("👉 *建议：加深/减淡字体颜色*")
                            if t.bg_noise > 0.2:
                                st.write("👉 *建议：添加文字底色块*")
            else:
                st.info("未检测到明显文字。")
    except Exception as e:
        st.error(f"无法打开图片，文件可能已损坏。错误信息: {e}")
else:
    st.info("👈 请先上传一张图片以开始分析。")