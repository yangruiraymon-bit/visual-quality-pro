import streamlit as st
import cv2
import numpy as np
from rembg import remove
from PIL import Image

def analyze_rule_of_thirds(image_pil):
    img_np = np.array(image_pil.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    vis_img = img_bgr.copy()
    w3, h3 = int(w/3), int(h/3)
    targets = [(w3, h3), (2*w3, h3), (w3, 2*h3), (2*w3, 2*h3)]
    target_names = ["左上点", "右上点", "左下点", "右下点"]
    grid_color = (200, 200, 200)
    cv2.line(vis_img, (w3, 0), (w3, h), grid_color, 1, cv2.LINE_AA)
    cv2.line(vis_img, (2*w3, 0), (2*w3, h), grid_color, 1, cv2.LINE_AA)
    cv2.line(vis_img, (0, h3), (w, h3), grid_color, 1, cv2.LINE_AA)
    cv2.line(vis_img, (0, 2*h3), (w, 2*h3), grid_color, 1, cv2.LINE_AA)
    for tx, ty in targets:
        cv2.circle(vis_img, (tx, ty), 6, (0, 215, 255), -1)
    mask_rgba = remove(img_np, alpha_matting=True)
    mask = mask_rgba[:, :, 3]
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    result_data = None
    if contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        if area > (w * h * 0.01):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                min_dist = float('inf')
                best_idx = -1
                for idx, (tx, ty) in enumerate(targets):
                    dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                diag_len = np.sqrt(w**2 + h**2)
                norm_threshold = diag_len / 6.0
                score = max(0, 100 * (1 - (min_dist / norm_threshold)))
                score_color = (0, 255, 0) if score > 80 else (0, 165, 255) if score > 50 else (0, 0, 255)
                cv2.drawContours(vis_img, [cnt], -1, score_color, 2)
                cv2.circle(vis_img, (cx, cy), 8, (0, 0, 255), -1)
                tx, ty = targets[best_idx]
                cv2.line(vis_img, (cx, cy), (tx, ty), (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(vis_img, f"{score:.0f}", (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)
                result_data = {"score": score, "target": target_names[best_idx], "distance": min_dist, "cx": cx, "cy": cy}
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    return vis_img, result_data

st.title("📸 AI 摄影构图：三分法则分析")
st.markdown("""
检测图像主体是否符合 **“井字构图” (Rule of Thirds)**。
系统计算**主体质心**到四个**黄金交点**的欧氏距离，距离越近，得分越高。
""")

with st.sidebar:
    st.header("🖼️ 图像上传")
    uploaded_file = st.file_uploader("选择一张照片...", type=['jpg', 'jpeg', 'png'])
    st.info("💡 **提示**\n黄色点 = 黄金分割点\n紫色线 = 距离偏差\n红色点 = 主体质心")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图片")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("分析结果")
        with st.spinner('正在进行 AI 显著性分割与几何计算...'):
            result_img, data = analyze_rule_of_thirds(image)
            st.image(result_img, use_container_width=True)
    if data:
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("构图评分", f"{data['score']:.1f} / 100", delta="完美" if data['score']>85 else "一般" if data['score']>60 else None)
        m2.metric("最近黄金点", data['target'])
        m3.metric("欧氏距离偏差", f"{data['distance']:.1f} px", delta="-越低越好")
        st.markdown("### 📝 AI 评价")
        if data['score'] >= 85:
            st.success(f"**极佳的构图！** 主体精准地落在了 **{data['target']}** 附近，视觉重心非常舒适。")
        elif data['score'] >= 60:
            st.info(f"**符合规范。** 主体靠近 {data['target']}，遵循了三分法原则。")
        else:
            st.warning("**居中或偏离。** 主体未落在三分线交点上。这可能是居中构图，或者需要进行二次裁剪。")
    else:
        st.warning("未检测到明显的主体，请尝试更换背景更干净的图片。")
else:
    st.info("👈 请在左侧上传图片开始分析")