import streamlit as st
import cv2
import numpy as np
from PIL import Image

# === 嵌入上面的类代码 ===
import cv2
import numpy as np

class ColorSymmetryAnalyzer:
    def __init__(self):
        self.MAX_RGB_DIST = np.sqrt(3 * (255**2))

    def analyze(self, image_input: np.ndarray):
        target_width = 512
        h, w = image_input.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        img_resized = cv2.resize(image_input, (target_width, new_h))
        
        h, w = img_resized.shape[:2]
        cx = w // 2
        left_half = img_resized[:, :cx]
        right_half = img_resized[:, cx:]
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]
        right_half_flipped = cv2.flip(right_half, 1)
        left_f = left_half.astype(np.float32)
        right_f = right_half_flipped.astype(np.float32)
        diff_map = np.linalg.norm(left_f - right_f, axis=2)
        med = np.mean(diff_map)
        sensitivity = 1.0
        normalized_diff = med / self.MAX_RGB_DIST
        score = max(0, 100 * (1 - normalized_diff * sensitivity))
        heatmap_norm = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        return {
            "score": round(score, 2),
            "med": round(med, 2),
            "heatmap": heatmap_color,
            "diff_map_raw": diff_map
        }

st.set_page_config(page_title="色彩对称性分析", layout="wide")

st.title("🎨 像素级分析：色彩对称性")
st.markdown("""
通过计算垂直中线两侧像素的 **RGB 欧氏距离**，量化画面的色彩镜像程度。
- **高分**：类似万花筒或镜面反射，左右颜色完全一致。
- **差异热力图**：红色越深，代表该区域左右不对称越严重。
""")

uploaded_file = st.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image_pil = Image.open(uploaded_file)
    img_np = np.array(image_pil.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    analyzer = ColorSymmetryAnalyzer()
    result = analyzer.analyze(img_bgr)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("原始图片")
        st.image(image_pil, use_container_width=True)

    with col2:
        st.subheader("不对称热力图")
        # 热力图越红，说明该位置左右差异越大
        st.image(cv2.cvtColor(result['heatmap'], cv2.COLOR_BGR2RGB), 
                 caption="红色=差异大，蓝色=差异小", use_container_width=True)

    with col3:
        st.subheader("分析报告")
        st.metric("色彩对称评分", f"{result['score']} / 100", 
                  delta="完美镜像" if result['score'] > 95 else None)
        
        st.metric("平均像素色差 (MED)", f"{result['med']}", 
                  delta="-越低越好", delta_color="inverse")
        
        st.info("💡 **解读**\n如果评分低但热力图显示只有局部是红色，说明大部分区域是对称的，只有局部物体破坏了对称性。")