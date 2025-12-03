import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional, Dict
import platform

# === 数据结构 ===
@dataclass
class OmniReport:
    # 构图维度
    composition_diagonal: float
    composition_thirds: float
    composition_balance: float
    composition_symmetry: float
    
    # 色彩/氛围维度
    color_warmth: float
    color_saturation: float
    color_brightness: float
    color_contrast: float
    color_clarity: float
    
    # 图底关系维度
    fg_area_diff: float
    fg_color_diff: float
    fg_texture_diff: float
    
    # 文字分析维度
    fg_text_present: bool
    fg_text_legibility: float
    fg_text_contrast: float

    # 可视化图像数据
    vis_mask: Optional[np.ndarray] = None
    vis_edge_fg: Optional[np.ndarray] = None
    vis_edge_bg: Optional[np.ndarray] = None
    vis_edge_composite: Optional[np.ndarray] = None
    vis_text_analysis: Optional[np.ndarray] = None
    vis_color_contrast: Optional[np.ndarray] = None
    vis_symmetry_heatmap: Optional[np.ndarray] = None
    vis_diag: Optional[np.ndarray] = None
    vis_thirds: Optional[np.ndarray] = None
    vis_balance: Optional[np.ndarray] = None
    vis_clarity: Optional[np.ndarray] = None
    vis_warmth: Optional[np.ndarray] = None
    vis_saturation: Optional[np.ndarray] = None
    vis_brightness: Optional[np.ndarray] = None
    vis_contrast: Optional[np.ndarray] = None

# === 混合分割器 ===
class HybridSegmenter:
    def __init__(self):
        print("🚀 Initializing Hybrid Segmenter...")
        try:
            self.semantic_model = YOLO("yolov8m-seg.pt")
        except Exception as e:
            print(f"YOLO load failed: {e}")
            self.semantic_model = None

    def _get_kmeans_candidates(self, img: np.ndarray, k: int = 5):
        h, w = img.shape[:2]
        try:
            blurred = cv2.pyrMeanShiftFiltering(img, 15, 30)
        except:
            blurred = cv2.GaussianBlur(img, (11, 11), 0)
            
        pixel_values = blurred.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.reshape((h, w))
        
        candidates = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        for i in range(k):
            mask = (labels == i).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            num, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for j in range(1, num):
                area = stats[j, cv2.CC_STAT_AREA]
                if area < (h * w * 0.02): continue
                
                comp_mask = (labels_im == j).astype(np.uint8) * 255
                cx, cy = centroids[j]
                dist_to_center = np.sqrt(((cx - w/2)/w)**2 + ((cy - h/2)/h)**2)
                
                # 简单检测是否接触边界
                touches_border = (np.any(comp_mask[0, :]) or np.any(comp_mask[-1, :]) or 
                                  np.any(comp_mask[:, 0]) or np.any(comp_mask[:, -1]))
                
                candidates.append({
                    "mask": comp_mask, 
                    "area": area, 
                    "border": touches_border,
                    "dist": dist_to_center
                })
        return candidates

    def extract_mask(self, image_bgr: np.ndarray, config: Dict) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        
        # 参数读取
        overlap_thresh = config.get('seg_iou_threshold', 0.3)
        kmeans_k = config.get('seg_kmeans_k', 5)

        # 1. 色块候选
        color_candidates = self._get_kmeans_candidates(image_bgr, k=kmeans_k)
        if not color_candidates:
            return np.zeros((h, w), dtype=np.uint8)

        # 2. 语义对象
        semantic_masks = []
        if self.semantic_model:
            results = self.semantic_model(image_bgr, verbose=False, retina_masks=True)
            if results[0].masks:
                for mask_tensor in results[0].masks.data:
                    m = mask_tensor.cpu().numpy()
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h))
                    bin_m = (m > 0.5).astype(np.uint8) * 255
                    semantic_masks.append(bin_m)

        # 3. 融合逻辑
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        if semantic_masks:
            for sem_mask in semantic_masks:
                # 语义对象优先，不再强求交集，而是直接合并
                # 除非我们想严格执行"视觉引导语义"
                # 这里根据之前的逻辑：YOLO 判定了就是主体
                final_mask = cv2.bitwise_or(final_mask, sem_mask)
            return final_mask
        else:
            # Fallback
            best_score = -1
            best_candidate = None
            for c in color_candidates:
                area_ratio = c['area'] / (h * w)
                if area_ratio > 0.9: continue
                border_penalty = 0.3 if c['border'] else 1.0
                center_bonus = 1.0 - c['dist']
                score = area_ratio * border_penalty * center_bonus
                if score > best_score:
                    best_score = score
                    best_candidate = c['mask']
            
            if best_candidate is not None:
                return best_candidate
            else:
                color_candidates.sort(key=lambda x: x['area'], reverse=True)
                return color_candidates[0]['mask']

# === 全能视觉分析引擎 ===
class OmniVisualEngine:
    def __init__(self):
        print("Initializing OCR Engine...")
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        self.segmenter = HybridSegmenter()

    def _load_safe_font(self, font_size=16):
        system = platform.system()
        font_path = None
        if system == "Windows": font_path = "C:/Windows/Fonts/msyh.ttc"
        elif system == "Darwin": font_path = "/System/Library/Fonts/PingFang.ttc"
        elif system == "Linux": font_path = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
        try: return ImageFont.truetype("font.ttf", font_size)
        except: pass
        if font_path:
            try: return ImageFont.truetype(font_path, font_size)
            except: pass
        return ImageFont.load_default()

    def analyze(self, image_input: np.ndarray, config: Dict = None) -> OmniReport:
        # === 0. 参数初始化 ===
        if config is None: config = {}
        
        # 提取配置 (设定默认值)
        process_w = config.get('process_width', 512)
        
        # 构图阈值
        th_diag = config.get('comp_diag_slope', 0.3)
        th_thirds = config.get('comp_thirds_slope', 0.2)
        th_sym = config.get('comp_sym_tolerance', 120.0) # 容差 120
        k_sym_blur = int(config.get('comp_sym_blur_k', 31))
        
        # 色彩阈值
        th_clarity = config.get('color_clarity_thresh', 0.7) # 亮度 > 0.7
        
        # 图底阈值
        ref_tex = config.get('fg_tex_norm', 50.0)
        
        # 文字阈值
        th_text_score = config.get('text_score_thresh', 60.0)

        # --- 1. 全局预处理 ---
        h, w = image_input.shape[:2]
        scale = process_w / w
        new_h = int(h * scale)
        img_small = cv2.resize(image_input, (process_w, new_h))
        
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        img_luv = cv2.cvtColor(img_small, cv2.COLOR_BGR2Luv)
        
        # --- 2. AI 主体分割 ---
        binary_mask = self.segmenter.extract_mask(img_small, config)
        binary_mask_inv = cv2.bitwise_not(binary_mask)
        
        # 初始化图
        vis_diag = img_small.copy()
        vis_thirds = img_small.copy()
        vis_balance = img_small.copy()
        
        score_diag, score_thirds, score_balance = 0.0, 0.0, 0.0
        
        # --- 3. 模块 A: 构图分析 ---
        M_global = cv2.moments(binary_mask)
        if M_global["m00"] > 0:
            cx = int(M_global["m10"] / M_global["m00"])
            cy = int(M_global["m01"] / M_global["m00"])
            
            # A1. 对角线
            diag_len = np.sqrt(new_h**2 + process_w**2)
            cv2.line(vis_diag, (0, 0), (process_w, new_h), (200, 200, 200), 1)
            cv2.line(vis_diag, (0, new_h), (process_w, 0), (200, 200, 200), 1)
            
            d1 = abs(new_h*cx - process_w*cy) / diag_len
            d2 = abs(new_h*cx + process_w*cy - process_w*new_h) / diag_len
            score_diag = max(0, 100 * (1 - min(d1, d2) / (diag_len * th_diag)))
            
            cv2.circle(vis_diag, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(vis_diag, f"Diag: {int(score_diag)}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # A2. 三分法
            w3, h3 = int(process_w/3), int(new_h/3)
            # 绘图略... (保持原样)
            grid_color = (0, 255, 255)
            cv2.line(vis_thirds, (w3, 0), (w3, new_h), grid_color, 1)
            cv2.line(vis_thirds, (2*w3, 0), (2*w3, new_h), grid_color, 1)
            cv2.line(vis_thirds, (0, h3), (process_w, h3), grid_color, 1)
            cv2.line(vis_thirds, (0, 2*h3), (process_w, 2*h3), grid_color, 1)
            
            targets = [(w3, h3), (2*w3, h3), (w3, 2*h3), (2*w3, 2*h3)]
            dists = [np.sqrt((cx-tx)**2 + (cy-ty)**2) for tx, ty in targets]
            score_thirds = max(0, 100 * (1 - min(dists) / (diag_len * th_thirds)))
            
            best_idx = dists.index(min(dists))
            tx, ty = targets[best_idx]
            cv2.circle(vis_thirds, (cx, cy), 6, (0, 0, 255), -1)
            cv2.line(vis_thirds, (cx, cy), (tx, ty), (0, 255, 0), 2)

            # A3. 平衡
            center_x = process_w // 2
            cv2.line(vis_balance, (center_x, 0), (center_x, new_h), (255, 255, 0), 2)
            m_l = cv2.countNonZero(binary_mask[:, :center_x])
            m_r = cv2.countNonZero(binary_mask[:, center_x:])
            if m_l + m_r > 0:
                score_balance = 100 * (1 - abs(m_l - m_r) / max(m_l, m_r))
                # 绘制略... (保持原样)
                ML, MR = cv2.moments(binary_mask[:, :center_x]), cv2.moments(binary_mask[:, center_x:])
                if ML["m00"] > 0:
                    cxl, cyl = int(ML["m10"]/ML["m00"]), int(ML["m01"]/ML["m00"])
                    cv2.circle(vis_balance, (cxl, cyl), 5, (0,255,0), -1)
                    cv2.line(vis_balance, (cxl, cyl), (center_x, cyl), (0,255,0), 2)
                if MR["m00"] > 0:
                    cxr, cyr = int(MR["m10"]/MR["m00"]) + center_x, int(MR["m01"]/MR["m00"])
                    cv2.circle(vis_balance, (cxr, cyr), 5, (0,0,255), -1)
                    cv2.line(vis_balance, (center_x, cyr), (cxr, cyr), (0,0,255), 2)

        # A4. 视觉稳定性 (RGB 欧氏距离) + 高斯模糊预处理
        try:
            if k_sym_blur % 2 == 0:
                k_sym_blur += 1
            img_blur = cv2.GaussianBlur(img_small, (k_sym_blur, k_sym_blur), 0)
            cx_sym = process_w // 2
            left_half = img_blur[:, :cx_sym]
            right_half = img_blur[:, -cx_sym:]
            right_flipped = cv2.flip(right_half, 1)
            diff_map = np.linalg.norm(left_half.astype(np.float32) - right_flipped.astype(np.float32), axis=2)
            mean_diff = float(np.mean(diff_map))
            score_symmetry = max(0.0, 100.0 * (1.0 - mean_diff / float(th_sym)))
            diff_full = np.hstack((diff_map, cv2.flip(diff_map, 1)))
            heatmap_norm = cv2.normalize(diff_full, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis_symmetry_heatmap = cv2.cvtColor(cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Symmetry error: {e}")
            score_symmetry = 0.0
            vis_symmetry_heatmap = None

        # --- 4. 模块 B: 色彩 (Luv) ---
        L, u, v = cv2.split(img_luv)
        L_float = L.astype(np.float32) / 255.0 # 归一化修复
        u_float = u.astype(np.float32) - 128.0
        v_float = v.astype(np.float32) - 128.0
        
        chroma = np.sqrt(u_float**2 + v_float**2)
        sat_mean = min(1.0, float(np.mean(chroma) / 128.0))
        chroma_vis = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_saturation = cv2.cvtColor(cv2.applyColorMap(chroma_vis, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        
        bri_mean = float(np.mean(L_float))
        vis_brightness = cv2.cvtColor(L, cv2.COLOR_GRAY2RGB)
        
        cont_std = float(np.std(L_float))
        vis_contrast = np.zeros_like(L)
        vis_contrast[L < 85] = 0
        vis_contrast[(L >= 85) & (L < 170)] = 127
        vis_contrast[L >= 170] = 255
        vis_contrast = cv2.cvtColor(vis_contrast, cv2.COLOR_GRAY2RGB)
        
        clarity_mask = (L_float >= th_clarity) # 使用配置阈值
        clarity_ratio = float(np.count_nonzero(clarity_mask) / L_float.size)
        vis_clarity = img_small.copy()
        vis_clarity[~clarity_mask] = (vis_clarity[~clarity_mask] * 0.3).astype(np.uint8)
        
        warm_mask = (v_float > 0)
        warmth_ratio = float(np.count_nonzero(warm_mask) / v.size)
        vis_warmth = np.zeros_like(img_small)
        vis_warmth[warm_mask] = [0, 0, 255]
        vis_warmth[~warm_mask] = [255, 0, 0]
        vis_warmth = cv2.cvtColor(vis_warmth, cv2.COLOR_BGR2RGB)

        # --- 5. 模块 C: 图底 ---
        total_px = binary_mask.size
        fg_px = cv2.countNonZero(binary_mask)
        area_diff = abs((fg_px/total_px) - (1 - fg_px/total_px)) if total_px > 0 else 0
        
        color_diff, texture_diff = 0.0, 0.0
        vis_edge_fg, vis_edge_bg, vis_edge_composite, vis_color_contrast = None, None, None, None

        if fg_px > 0 and (total_px - fg_px) > 0:
            m_fg = cv2.mean(img_luv, mask=binary_mask)[:3]
            m_bg = cv2.mean(img_luv, mask=binary_mask_inv)[:3]
            fg_v = np.array([m_fg[0], m_fg[1]-128, m_fg[2]-128])
            bg_v = np.array([m_bg[0], m_bg[1]-128, m_bg[2]-128])
            color_diff = float(np.linalg.norm(fg_v - bg_v))
            
            mean_fg_bgr = cv2.cvtColor(np.array([[[m_fg[0], m_fg[1], m_fg[2]]]], dtype=np.uint8), cv2.COLOR_Luv2BGR)[0][0].tolist()
            mean_bg_bgr = cv2.cvtColor(np.array([[[m_bg[0], m_bg[1], m_bg[2]]]], dtype=np.uint8), cv2.COLOR_Luv2BGR)[0][0].tolist()
            contrast_canvas = np.zeros((300, 300, 3), dtype=np.uint8)
            contrast_canvas[:] = mean_bg_bgr
            cv2.circle(contrast_canvas, (150, 150), 100, mean_fg_bgr, -1)
            vis_color_contrast = cv2.cvtColor(contrast_canvas, cv2.COLOR_BGR2RGB)
            
            # Sobel 纹理
            grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(grad_x, grad_y)
            tex_fg = np.mean(mag[binary_mask > 0])
            tex_bg = np.mean(mag[binary_mask_inv > 0])
            texture_diff = min(1.0, abs(tex_fg - tex_bg) / ref_tex) # 使用配置参数
            
            mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            edge_fg = cv2.bitwise_and(mag_vis, mag_vis, mask=binary_mask)
            edge_bg = cv2.bitwise_and(mag_vis, mag_vis, mask=binary_mask_inv)
            composite = np.zeros_like(img_small, dtype=np.uint8)
            composite[:, :, 0] = edge_bg
            composite[:, :, 1] = edge_fg
            vis_edge_composite = composite

        # --- 6. 模块 D: 文字 ---
        vis_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(vis_pil)
        font = self._load_safe_font(16)
        
        ocr_results = self.reader.readtext(img_small)
        text_scores = []
        text_contrasts = []
        
        for (bbox, text_content, prob) in ocr_results:
            pts = np.array(bbox, dtype=np.int32)
            x, y, w_box, h_box = cv2.boundingRect(pts)
            x, y = max(0, x), max(0, y)
            w_box = min(w_box, process_w - x)
            h_box = min(h_box, new_h - y)
            if w_box < 5 or h_box < 5: continue
            
            roi_g = img_gray[y:y+h_box, x:x+w_box]
            roi_c = img_small[y:y+h_box, x:x+w_box]
            
            try:
                _, t_mask = cv2.threshold(roi_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if np.mean([t_mask[0,0], t_mask[-1,-1]]) > 127: t_mask = cv2.bitwise_not(t_mask)
                b_mask = cv2.bitwise_not(t_mask)
                
                m_txt = cv2.mean(roi_c, mask=t_mask)[:3]
                m_bg = cv2.mean(roi_c, mask=b_mask)[:3]
                contrast = float(np.linalg.norm(np.array(m_txt) - np.array(m_bg)))
                
                # 评分
                s_con = min(100, contrast)
                item_score = 0.7 * s_con + 0.3 * 80 
                
                text_scores.append(item_score)
                text_contrasts.append(contrast)
                
                is_legible = item_score > th_text_score # 使用配置阈值
                color = (0, 255, 0) if is_legible else (255, 0, 0)
                draw.rectangle([x, y, x+w_box, y+h_box], outline=color, width=2)
                
                label = f"{text_content} | {int(item_score)}"
                bbox_txt = draw.textbbox((x, y), label, font=font)
                draw.rectangle(bbox_txt, fill=color)
                draw.text((x, y), label, fill=(255, 255, 255), font=font)
            except: continue
                
        has_text = len(text_scores) > 0
        avg_text_score = float(np.mean(text_scores)) if has_text else 0.0
        avg_text_contrast = float(np.mean(text_contrasts)) if has_text else 0.0
        vis_text_final = np.array(vis_pil)

        return OmniReport(
            composition_diagonal=round(score_diag, 1),
            composition_thirds=round(score_thirds, 1),
            composition_balance=round(score_balance, 1),
            composition_symmetry=round(score_symmetry, 1),
            color_warmth=round(warmth_ratio, 2),
            color_saturation=round(sat_mean, 2),
            color_brightness=round(bri_mean, 2),
            color_contrast=round(cont_std, 2),
            color_clarity=round(clarity_ratio, 2),
            fg_area_diff=round(area_diff, 2),
            fg_color_diff=round(color_diff, 1),
            fg_texture_diff=round(texture_diff, 3),
            fg_text_present=has_text,
            fg_text_legibility=round(avg_text_score, 1),
            fg_text_contrast=round(avg_text_contrast, 1),
            
            vis_mask=binary_mask,
            vis_edge_fg=vis_edge_fg,
            vis_edge_bg=vis_edge_bg,
            vis_edge_composite=vis_edge_composite,
            vis_text_analysis=vis_text_final,
            vis_color_contrast=vis_color_contrast,
            vis_symmetry_heatmap=vis_symmetry_heatmap,
            vis_diag=cv2.cvtColor(vis_diag, cv2.COLOR_BGR2RGB),
            vis_thirds=cv2.cvtColor(vis_thirds, cv2.COLOR_BGR2RGB),
            vis_balance=cv2.cvtColor(vis_balance, cv2.COLOR_BGR2RGB),
            vis_clarity=cv2.cvtColor(vis_clarity, cv2.COLOR_BGR2RGB),
            vis_warmth=vis_warmth,
            vis_saturation=vis_saturation,
            vis_brightness=vis_brightness,
            vis_contrast=vis_contrast
        )

# === 诊断器 (支持权重配置) ===
class AestheticDiagnostician:
    @staticmethod
    def generate_report(data: OmniReport, config: Dict = None) -> dict:
        if config is None: config = {}
        
        # 权重读取
        w_comp = config.get('weight_composition', 0.3)
        w_color = config.get('weight_color', 0.3)
        w_fg = config.get('weight_figure_ground', 0.4)
        
        # 构图分
        comp_score = (data.composition_diagonal + data.composition_thirds + 
                      data.composition_balance + data.composition_symmetry) / 4
        
        color_score = (data.color_clarity * 100 * 0.4 + 
                       data.color_saturation * 100 * 0.3 + 
                       data.color_contrast * 100 * 0.3)
        
        # 图底分
        if data.fg_text_present:
            fg_score = data.fg_text_legibility * 0.6 + min(100, data.fg_color_diff) * 0.4
        else:
            fg_score = min(100, data.fg_color_diff) * 0.6 + data.fg_area_diff * 100 * 0.4
            
        if data.color_clarity > 0.85:
            color_score = color_score * 0.6

        total_score = int(w_comp * comp_score + w_color * color_score + w_fg * fg_score)

        # 评级
        if total_score >= 85: rating = "S (大师级)"
        elif total_score >= 70: rating = "A (优秀)"
        elif total_score >= 60: rating = "B (合格)"
        else: rating = "C (需改进)"

        # 标签 (略微简化逻辑)
        tags = []
        if data.color_warmth > 0.6: tags.append("暖色调")
        elif data.color_warmth < 0.3: tags.append("冷色调")
        if data.color_saturation > 0.5: tags.append("高饱和")
        if data.color_brightness > 0.6: tags.append("高调")
        elif data.color_brightness < 0.3: tags.append("低调")
        if data.composition_diagonal > 80: tags.append("动感构图")
        
        # 优缺点
        pros, cons, suggestions = [], [], []
        
        if data.composition_symmetry > 90: pros.append("极佳的视觉秩序感")
        if data.fg_color_diff > 100: pros.append("主体色彩醒目")
        if data.color_clarity > 0.85:
            cons.append("高光溢出严重（过曝），亮部细节大量丢失。")
            suggestions.append("调色建议：降低曝光度或高光，找回亮部细节。")
        elif data.color_clarity > 0.3:
            pros.append("光影通透，高光区域充足，视觉传达效率高。")
        elif data.color_clarity < 0.1:
            cons.append("画面整体灰暗沉闷，存在明显的‘雾霾感’。")
            suggestions.append("调色建议：提升高光亮度或使用去雾工具，增加画面通透感。")
        
        
        if data.composition_balance < 40:
            cons.append("物理重心失衡")
            suggestions.append("调整主体位置平衡左右")
        if data.fg_text_present and data.fg_text_legibility < 60:
            cons.append("部分文字难辨认")
            suggestions.append("给文字添加底板或描边")

        return {
            "total_score": total_score,
            "rating_level": rating,
            "style_tags": tags,
            "summary": "AI 分析完成。",
            "pros": pros,
            "cons": cons,
            "suggestions": suggestions
        }