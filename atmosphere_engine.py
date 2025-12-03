import streamlit as st
import cv2
import numpy as np
from PIL import Image
# 假设上面的类定义在 atmosphere_engine.py 中
# from atmosphere_engine import AtmosphereAnalyzer

# === 简单的内置类 (为了演示方便) ===
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class AtmosphereMetrics:
    warmth_ratio: float      # 暖色占比 (0-1)
    mean_saturation: float   # 平均饱和度 (0-1)
    mean_brightness: float   # 平均亮度 (0-1)
    brightness_std: float    # 亮度对比度 (标准差)
    clarity_ratio: float     # 清晰度 (0-1)

class AtmosphereAnalyzer:
    def __init__(self):
        # 冷色调范围定义 (OpenCV Hue: 0-179)
        # 30 (Green-Yellow) ~ 110 (Blue)
        self.cool_h_min = 30
        self.cool_h_max = 110
        
        # 清晰度亮度阈值 (0.7 - 1.0)
        self.clarity_thresh_low = 0.7 * 255

    def analyze(self, image_input: np.ndarray) -> AtmosphereMetrics:
        """
        :param image_input: OpenCV BGR 格式图像
        """
        # 1. 转换色彩空间 BGR -> HSV
        hsv = cv2.cvtColor(image_input, cv2.COLOR_BGR2HSV)
        
        # 分离通道，并转为 float 进行计算
        h = hsv[:, :, 0]
        s = hsv[:, :, 1].astype(np.float32) / 255.0  # 归一化到 0-1
        v = hsv[:, :, 2].astype(np.float32)          # 保持 0-255 计算阈值，后续计算均值时归一化
        
        total_pixels = h.size

        # --- A. 暖色调主导 ---
        # 冷色 mask: 30 <= H <= 110
        cool_mask = (h >= self.cool_h_min) & (h <= self.cool_h_max)
        # 暖色像素数 = 总数 - 冷色数 (或者直接取反)
        warm_pixels = total_pixels - np.count_nonzero(cool_mask)
        warmth_ratio = warm_pixels / total_pixels

        # --- B. 饱和度 ---
        mean_saturation = np.mean(s)

        # --- C. 亮度 ---
        # v 目前是 0-255
        mean_brightness = np.mean(v) / 255.0

        # --- D. 亮度对比度 (标准差) ---
        # 标准差除以 255 归一化，以便于理解
        brightness_std = np.std(v) / 255.0

        # --- E. 清晰度 ---
        # 统计 v 在 [0.7*255, 255] 范围内的像素
        clarity_mask = (v >= self.clarity_thresh_low)
        clarity_ratio = np.count_nonzero(clarity_mask) / total_pixels

        return AtmosphereMetrics(
            warmth_ratio=float(round(warmth_ratio, 3)),
            mean_saturation=float(round(mean_saturation, 3)),
            mean_brightness=float(round(mean_brightness, 3)),
            brightness_std=float(round(brightness_std, 3)),
            clarity_ratio=float(round(clarity_ratio, 3))
        )

st.set_page_config(page_title="色彩情绪分析", layout="wide", page_icon="🎨")

st.title("🎨 图像氛围与情绪分析")
st.markdown("""
基于 HSV 色彩心理学模型，量化图像的 **兴奋度、愉悦感、光照分布及清晰度**。
""")

uploaded_file = st.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image_pil = Image.open(uploaded_file)
    img_np = np.array(image_pil.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    analyzer = AtmosphereAnalyzer()
    metrics = analyzer.analyze(img_bgr)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("原始图片")
        st.image(image_pil, use_container_width=True)

    with col2:
        st.subheader("氛围指标分析")
        
        # 1. 暖色调 (Warmth)
        st.write("#### 🔥 暖色调占比 (Warmth)")
        st.progress(metrics.warmth_ratio)
        if metrics.warmth_ratio > 0.5:
            st.caption(f"占比 {metrics.warmth_ratio:.0%}: **暖色主导 (兴奋/活力)** - 红黄色系为主")
        else:
            st.caption(f"占比 {metrics.warmth_ratio:.0%}: **冷色主导 (放松/冷静)** - 蓝绿色系为主")

        # 2. 饱和度 (Saturation)
        st.write("#### 🌈 饱和度 (Saturation)")
        st.progress(metrics.mean_saturation)
        emotion = "快乐/纯洁" if metrics.mean_saturation > 0.4 else "低沉/悲伤"
        st.caption(f"均值 {metrics.mean_saturation:.2f}: **{emotion}** - 色彩鲜艳度")

        # 3. 亮度 (Brightness)
        st.write("#### ☀️ 亮度 (Brightness)")
        st.progress(metrics.mean_brightness)
        st.caption(f"均值 {metrics.mean_brightness:.2f}: **信息传递效率** - 越亮越清晰")

        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            # 4. 亮度对比度 (Contrast)
            st.metric("🌗 亮度对比度", f"{metrics.brightness_std:.2f}", 
                      delta="- 越低越柔和", delta_color="inverse")
            st.caption("低值=光照均匀(柔和)\n高值=光影生硬(戏剧性)")
            
        with col_sub2:
            # 5. 清晰度 (Clarity)
            st.metric("✨ 清晰度/去雾", f"{metrics.clarity_ratio:.0%}")
            st.caption("高亮像素占比\n值越高越透亮，无雾霾感")

    # 综合评价
    st.markdown("---")
    st.subheader("📝 综合心理学解读")
    
    analysis_text = []
    
    # 情绪倾向
    if metrics.warmth_ratio > 0.6 and metrics.mean_saturation > 0.5:
        analysis_text.append("🔥 **高兴奋度图像**：暖色且鲜艳，适合表达激情、促销或活力场景。")
    elif metrics.warmth_ratio < 0.4 and metrics.mean_brightness > 0.6:
        analysis_text.append("🍃 **高放松度图像**：冷色且明亮，适合表达医疗、科技或宁静的自然场景。")
    
    # 质感倾向
    if metrics.brightness_std < 0.15:
        analysis_text.append("☁️ **柔光质感**：光照非常均匀，给人舒适、亲切的感受（如日系写真）。")
    elif metrics.brightness_std > 0.25:
        analysis_text.append("⚡ **硬朗质感**：光影对比强烈，具有较强的视觉冲击力。")
        
    if metrics.clarity_ratio < 0.1:
        analysis_text.append("🌫️ **朦胧感/雾霾**：清晰度较低，可能需要后期去雾或调整曝光。")

    for text in analysis_text:
        st.info(text)