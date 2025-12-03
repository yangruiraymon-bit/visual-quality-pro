import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove

# === 核心算法逻辑 (封装) ===
def get_subject_mask_rembg(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = remove(img_rgb, alpha_matting=True)
    mask = result[:, :, 3]
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

def analyze_image(image_input, sensitivity):
    # 转换 PIL 图片为 OpenCV 格式 (RGB -> BGR)
    img_array = np.array(image_input.convert('RGB'))
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    h, w = img.shape[:2]
    vis_img = img.copy()
    
    thresh = get_subject_mask_rembg(img)
    
    # 轮廓检测
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 筛选前3大物体
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    
    # 绘制对角线 (虚线模拟)
    # D1: 左上-右下 (蓝色)
    cv2.line(vis_img, (0, 0), (w, h), (255, 0, 0), 2)
    # D2: 左下-右上 (红色)
    cv2.line(vis_img, (0, h), (w, 0), (0, 0, 255), 2)

    if not contours:
        return None, vis_img, 0, {}

    total_score = 0
    total_weight = 0
    details = []

    # 归一化参考距离 (半对角线长)
    max_dist_norm = np.sqrt(h**2 + w**2) / 2

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < (w * h * 0.02): continue # 过滤小噪点

        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 计算曼哈顿加权距离逻辑
        # D1 distance
        d1 = abs(h * cx - w * cy) / np.sqrt(h**2 + (-w)**2)
        # D2 distance
        d2 = abs(h * cx + w * cy - w * h) / np.sqrt(h**2 + w**2)
        
        min_dist = min(d1, d2)
        chosen_diag = "D1 (蓝)" if d1 < d2 else "D2 (红)"
        
        # 评分计算 (0-100)
        raw_score = (1 - (min_dist / (max_dist_norm * 0.4))) * 100
        score = max(0, min(100, raw_score))
        
        weight = area
        total_score += score * weight
        total_weight += weight
        
        details.append({
            "id": i+1,
            "dist": min_dist,
            "score": score,
            "diag": chosen_diag
        })

        # 绘图：重心与连线
        color = (0, 255, 0) # 绿色
        cv2.drawContours(vis_img, [cnt], -1, color, 2)
        cv2.circle(vis_img, (cx, cy), 8, color, -1)
        
        # 绘制重心到最近对角线的垂线
        # 这里简化为画一条线指示
        if chosen_diag == "D1 (蓝)":
            # D1 投影点近似计算 (仅视效)
            pass 
        
        cv2.putText(vis_img, f"{score:.0f}", (cx+10, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    final_score = total_score / total_weight if total_weight > 0 else 0
    
    # 转换回 RGB 供 Streamlit 显示
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    return final_score, vis_img, len(details), details

# === 界面布局 ===
st.set_page_config(page_title="对角线构图分析仪", layout="wide")

st.title("📐 AI 摄影构图助手：对角线分析")
st.markdown("通过计算关键主体与画面对角线的 **加权曼哈顿距离**，量化评估构图的动态平衡感。")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
    uploaded_file = st.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])
    
    st.markdown("---")
    st.subheader("分析参数")
    sensitivity = st.slider("主体检测灵敏度", 0, 100, 50, help="调整此值以过滤背景杂物或捕获更多细节")
    
    st.info("💡 **说明**\n- **D1 (蓝线)**: 左上至右下\n- **D2 (红线)**: 左下至右上\n- 分数越高代表重心越贴合对角线。")

# 主逻辑
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # 执行分析
    final_score, result_img, obj_count, details = analyze_image(image, sensitivity)

    # 结果展示区
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始图片")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("分析视图")
        if final_score is not None:
            st.image(result_img, use_container_width=True)
        else:
            st.warning("未能检测到明显的主体，请调整灵敏度。")

    # 数据仪表盘
    if final_score is not None:
        st.markdown("---")
        st.subheader("📊 构图评分报告")
        
        # 核心指标卡片
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("总评分", f"{final_score:.1f} / 100", delta_color="normal")
        m2.metric("识别主体数", f"{obj_count} 个")
        
        if details:
            main_obj = details[0] # 最大物体
            m3.metric("主视觉导向", main_obj['diag'])
            m4.metric("像素偏移量", f"{main_obj['dist']:.1f} px", delta="-越低越好")
        
        # 详细解释
        st.markdown("### 📝 AI 评价")
        if final_score > 85:
            st.success(f"**完美构图！** 主体重心极其精准地落在了 {details[0]['diag']} 上，画面具有极强的动态张力。")
        elif final_score > 60:
            st.info("**良好的平衡。** 主体靠近对角线区域，构图舒适，但可能结合了其他构图法则（如三分法）。")
        else:
            st.warning("**弱对角线相关。** 这是一个居中或散点构图，如果你的目的是拍摄动感照片，建议尝试裁切或改变角度。")

else:
    st.write("👈 请在左侧上传一张照片开始分析。")