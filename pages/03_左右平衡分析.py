import streamlit as st
import cv2
import numpy as np
from PIL import Image
from balance_engine import SymmetryAnalyzer

st.title("⚖️ AI 构图分析：左右平衡强度")
st.markdown("""
量化分析图像沿**垂直中线**的对称性。
系统计算左右两半关键区域的**质心力臂 (Lever Arm)**，力臂长度越接近，对称性评分越高。
""")

with st.sidebar:
    st.header("控制台")
    uploaded_file = st.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])
    st.info("💡 **图示说明**\n- **青色线**: 画面中轴\n- **绿线**: 左侧力臂\n- **红线**: 右侧力臂\n- 长度越接近，得分越高")

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    analyzer = SymmetryAnalyzer()
    with st.spinner("正在计算左右质心力矩..."):
        result = analyzer.analyze(img_bgr)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("分析视图")
        st.image(cv2.cvtColor(result.visualization, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("平衡数据报告")
        st.metric("📏 距离对称性评分", f"{result.score} / 100", delta="完美对称" if result.score > 90 else None)
        st.write("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("左侧力臂", f"{result.left_arm} px")
        c2.metric("右侧力臂", f"{result.right_arm} px")
        diff = abs(result.left_arm - result.right_arm)
        c3.metric("偏差值", f"{diff} px", delta="-越低越好")
        st.write("---")
        st.markdown("#### ⚛️ 物理力矩平衡 (Visual Equilibrium)")
        st.write("考虑到左右物体的大小(面积)不同，真实的视觉平衡如下：")
        st.progress(int(result.equilibrium))
        st.caption(f"力矩平衡得分: {result.equilibrium} (结合了面积权重的综合平衡感)")
        if result.score > 85:
            st.success("⚖️ **高度对称！** 左右主体的视觉中心几乎与中轴线等距，画面极其稳定。")
        elif result.score > 60:
            st.info("⚖️ **基本平衡。** 左右存在轻微偏差，但整体构图稳定。")
        else:
            st.warning("⚖️ **非对称构图。** 画面重心明显偏向一侧，或两侧物体分布极不均匀。")
else:
    st.info("👈 请上传一张图片以检测平衡性")