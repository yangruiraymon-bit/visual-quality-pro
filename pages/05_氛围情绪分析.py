import streamlit as st
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass

@dataclass
class AtmosphereMetrics:
    warmth_ratio: float
    mean_saturation: float
    mean_brightness: float
    brightness_std: float
    clarity_ratio: float

class AtmosphereAnalyzer:
    def __init__(self):
        self.cool_h_min = 30
        self.cool_h_max = 110
        self.clarity_thresh_low = 0.7 * 255
    def analyze(self, image_input: np.ndarray) -> AtmosphereMetrics:
        hsv = cv2.cvtColor(image_input, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32)
        total_pixels = h.size
        cool_mask = (h >= self.cool_h_min) & (h <= self.cool_h_max)
        warm_pixels = total_pixels - np.count_nonzero(cool_mask)
        warmth_ratio = warm_pixels / total_pixels
        mean_saturation = np.mean(s)
        mean_brightness = np.mean(v) / 255.0
        brightness_std = np.std(v) / 255.0
        clarity_mask = (v >= self.clarity_thresh_low)
        clarity_ratio = np.count_nonzero(clarity_mask) / total_pixels
        return AtmosphereMetrics(
            warmth_ratio=float(round(warmth_ratio, 3)),
            mean_saturation=float(round(mean_saturation, 3)),
            mean_brightness=float(round(mean_brightness, 3)),
            brightness_std=float(round(brightness_std, 3)),
            clarity_ratio=float(round(clarity_ratio, 3))
        )

st.title("🎨 图像氛围与情绪分析")
st.markdown("基于 HSV 色彩心理学模型，量化图像的 兴奋度、愉悦感、光照分布及清晰度。")

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
        st.write("#### 🔥 暖色调占比 (Warmth)")
        st.progress(metrics.warmth_ratio)
        if metrics.warmth_ratio > 0.5:
            st.caption(f"占比 {metrics.warmth_ratio:.0%}: 暖色主导 (兴奋/活力)")
        else:
            st.caption(f"占比 {metrics.warmth_ratio:.0%}: 冷色主导 (放松/冷静)")
        st.write("#### 🌈 饱和度 (Saturation)")
        st.progress(metrics.mean_saturation)
        emotion = "快乐/纯洁" if metrics.mean_saturation > 0.4 else "低沉/悲伤"
        st.caption(f"均值 {metrics.mean_saturation:.2f}: {emotion}")
        st.write("#### ☀️ 亮度 (Brightness)")
        st.progress(metrics.mean_brightness)
        st.caption(f"均值 {metrics.mean_brightness:.2f}: 信息传递效率")
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            st.metric("🌗 亮度对比度", f"{metrics.brightness_std:.2f}", delta="- 越低越柔和", delta_color="inverse")
        with col_sub2:
            st.metric("✨ 清晰度/去雾", f"{metrics.clarity_ratio:.0%}")
    st.markdown("---")
    st.subheader("📝 综合心理学解读")
    analysis_text = []
    if metrics.warmth_ratio > 0.6 and metrics.mean_saturation > 0.5:
        analysis_text.append("🔥 高兴奋度图像：暖色且鲜艳，适合表达激情、促销或活力场景。")
    elif metrics.warmth_ratio < 0.4 and metrics.mean_brightness > 0.6:
        analysis_text.append("🍃 高放松度图像：冷色且明亮，适合表达医疗、科技或宁静的自然场景。")
    if metrics.brightness_std < 0.15:
        analysis_text.append("☁️ 柔光质感：光照非常均匀，给人舒适、亲切的感受。")
    elif metrics.brightness_std > 0.25:
        analysis_text.append("⚡ 硬朗质感：光影对比强烈，具有较强的视觉冲击力。")
    if metrics.clarity_ratio < 0.1:
        analysis_text.append("🌫️ 朦胧感/雾霾：清晰度较低，可能需要后期去雾或调整曝光。")
    for text in analysis_text:
        st.info(text)