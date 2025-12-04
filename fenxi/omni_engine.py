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
    fg_text_content: str

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

    def _get_background_mask(self, img: np.ndarray, k: int = 3) -> np.ndarray:
        h, w = img.shape[:2]
        scale = 0.5
        small_h, small_w = int(h * scale), int(w * scale)
        small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        data = small_img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.flatten()
        counts = np.bincount(labels)
        bg_label = np.argmax(counts)
        fg_mask_small = (labels != bg_label).astype(np.uint8) * 255
        fg_mask_small = fg_mask_small.reshape((small_h, small_w))
        fg_mask = cv2.resize(fg_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return fg_mask

    def _get_yolo_masks(self, img: np.ndarray) -> List[np.ndarray]:
        masks = []
        if self.semantic_model:
            results = self.semantic_model(img, verbose=False, retina_masks=True)
            if results[0].masks:
                h, w = img.shape[:2]
                for mask_tensor in results[0].masks.data:
                    m = mask_tensor.cpu().numpy()
                    if m.shape != (h, w): m = cv2.resize(m, (w, h))
                    bin_m = (m > 0.5).astype(np.uint8) * 255
                    masks.append(bin_m)
        return masks

    def extract_all_elements_mask(self, image_bgr: np.ndarray, config: Dict, text_boxes: List = None) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        base_mask = self._get_background_mask(image_bgr, k=3)
        text_mask = np.zeros((h, w), dtype=np.uint8)
        if text_boxes:
            for box in text_boxes:
                pts = np.array(box, dtype=np.int32)
                cv2.fillPoly(text_mask, [pts], 255)
        yolo_mask = np.zeros((h, w), dtype=np.uint8)
        yolo_results = self._get_yolo_masks(image_bgr)
        for m in yolo_results:
            yolo_mask = cv2.bitwise_or(yolo_mask, m)
        combined_mask = cv2.bitwise_or(base_mask, text_mask)
        combined_mask = cv2.bitwise_or(combined_mask, yolo_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return combined_mask

    def extract_main_subject_mask(self, image_bgr: np.ndarray, config: Dict, text_boxes: List = None) -> np.ndarray:
        return self._get_background_mask(image_bgr, k=5)

    def _get_kmeans_result(self, img: np.ndarray, k: int = 5) -> (np.ndarray, int):
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w//2), max(1, h//2)), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(small, (9, 9), 0)
        pixels = blurred.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        sh = small.shape[:2]
        labels = labels.reshape(sh)
        corner = [labels[0,0], labels[0,-1], labels[-1,0], labels[-1,-1]]
        bg = max(set(corner), key=corner.count)
        labels_full = cv2.resize(labels.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return labels_full, bg

    def _get_yolo_masks(self, img: np.ndarray) -> List[np.ndarray]:
        masks = []
        if self.semantic_model:
            results = self.semantic_model(img, verbose=False, retina_masks=True)
            if results and results[0].masks:
                h, w = img.shape[:2]
                for mask_tensor in results[0].masks.data:
                    m = mask_tensor.cpu().numpy()
                    if m.shape != (h, w): m = cv2.resize(m, (w, h))
                    masks.append(((m > 0.5).astype(np.uint8) * 255))
        return masks

    def extract_all_elements_mask(self, image_bgr: np.ndarray, config: Dict) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        yolo_masks = self._get_yolo_masks(image_bgr)
        if len(yolo_masks) > 0:
            combined = np.zeros((h, w), dtype=np.uint8)
            for m in yolo_masks:
                combined = cv2.bitwise_or(combined, m)
            return combined
        labels, bg_label = self._get_kmeans_result(image_bgr, config.get('seg_kmeans_k', 5))
        elements = (labels != bg_label).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        elements = cv2.morphologyEx(elements, cv2.MORPH_OPEN, kernel)
        return elements

# === 全能视觉分析引擎 ===
class OmniVisualEngine:
    def __init__(self):
        print("Initializing OCR Engine...")
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        self.segmenter = HybridSegmenter()
        
        # 尝试加载姿态模型用于辅助构图 (人脸重心)
        try:
            print("Loading Pose Model...")
            self.pose_model = YOLO("yolov8n-pose.pt")
        except:
            self.pose_model = None

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
        if config is None: config = {}
        
        process_w = config.get('process_width', 512)
        th_diag = config.get('comp_diag_slope', 0.3)
        th_thirds = config.get('comp_thirds_slope', 0.2)
        th_sym = config.get('comp_sym_tolerance', 120.0)
        th_sym_blur = config.get('comp_sym_blur_k', 31)
        th_clarity = config.get('color_clarity_thresh', 0.7)
        ref_tex = config.get('fg_tex_norm', 50.0)
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
        
        # --- 1.5 OCR ---
        ocr_raw = self.reader.readtext(img_small)
        text_boxes_for_seg = [item[0] for item in ocr_raw]
        
        # --- 1.6 人脸检测 (辅助) ---
        face_points = []
        if self.pose_model:
            try:
                pose_results = self.pose_model(img_small, verbose=False)
                if pose_results[0].keypoints is not None:
                    keypoints = pose_results[0].keypoints.data.cpu().numpy()
                    for person_kpts in keypoints:
                        nose = person_kpts[0] # 鼻子
                        if nose[2] > 0.5:
                            face_points.append((int(nose[0]), int(nose[1])))
            except: pass

        # --- 2. 分割 ---
        binary_mask = self.segmenter.extract_main_subject_mask(img_small, config, text_boxes_for_seg)
        binary_mask_inv = cv2.bitwise_not(binary_mask)
        
        vis_diag = img_small.copy()
        vis_thirds = img_small.copy()
        vis_balance = img_small.copy()
        
        score_diag, score_thirds, score_balance = 0.0, 0.0, 0.0
        
        # --- 3. 模块 A: 构图分析 ---
        
        # A1. 对角线
        M_global = cv2.moments(binary_mask)
        if M_global["m00"] > 0:
            cx = int(M_global["m10"] / M_global["m00"])
            cy = int(M_global["m01"] / M_global["m00"])
            
            diag_len = np.sqrt(new_h**2 + process_w**2)
            cv2.line(vis_diag, (0, 0), (process_w, new_h), (200, 200, 200), 1)
            cv2.line(vis_diag, (0, new_h), (process_w, 0), (200, 200, 200), 1)
            
            d1 = abs(new_h*cx - process_w*cy) / diag_len
            d2 = abs(new_h*cx + process_w*cy - process_w*new_h) / diag_len
            score_diag = max(0, 100 * (1 - min(d1, d2) / (diag_len * th_diag)))
            
            # [新增] 绘制主体的包围框
            x, y, bw, bh = cv2.boundingRect(binary_mask)
            cv2.rectangle(vis_diag, (x, y), (x+bw, y+bh), (0, 255, 0), 1)
            
            cv2.circle(vis_diag, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(vis_diag, f"Diag: {int(score_diag)}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # === A2. 三分法 (多元素+边框显示) ===
            w3, h3 = int(process_w/3), int(new_h/3)
            grid_color = (0, 255, 255)
            cv2.line(vis_thirds, (w3, 0), (w3, new_h), grid_color, 1)
            cv2.line(vis_thirds, (2*w3, 0), (2*w3, new_h), grid_color, 1)
            cv2.line(vis_thirds, (0, h3), (process_w, h3), grid_color, 1)
            cv2.line(vis_thirds, (0, 2*h3), (process_w, 2*h3), grid_color, 1)
            
            lines_x = [w3, 2*w3]
            lines_y = [h3, 2*h3]
            
            visual_elements = []
            
            # (1) 图形轮廓 (主体)
            sub_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sub_contours:
                area = cv2.contourArea(cnt)
                if area < (new_h * process_w * 0.005): continue
                
                # 计算包围框
                bx, by, bw, bh = cv2.boundingRect(cnt)
                
                M_sub = cv2.moments(cnt)
                if M_sub["m00"] == 0: continue
                geo_cx = int(M_sub["m10"] / M_sub["m00"])
                geo_cy = int(M_sub["m01"] / M_sub["m00"])
                
                # 人脸优先
                final_cx, final_cy = geo_cx, geo_cy
                is_face = False
                for fx, fy in face_points:
                    if cv2.pointPolygonTest(cnt, (fx, fy), True) > -20:
                        final_cx, final_cy = fx, fy
                        is_face = True
                        break
                
                visual_elements.append({
                    "type": "graphic",
                    "centroid": (final_cx, final_cy),
                    "bbox": (bx, by, bw, bh), # 存储bbox
                    "area": area,
                    "color": (255, 0, 255) if is_face else (0, 0, 255) # 紫/红
                })
            
            # (2) 文字元素
            for (bbox, text, prob) in ocr_raw:
                if prob < 0.3: continue
                pts = np.array(bbox, dtype=np.int32)
                bx, by, bw, bh = cv2.boundingRect(pts)
                
                M_txt = cv2.moments(pts)
                if M_txt["m00"] == 0: continue
                tcx = int(M_txt["m10"] / M_txt["m00"])
                tcy = int(M_txt["m01"] / M_txt["m00"])
                tarea = cv2.contourArea(pts)
                if tarea < (new_h * process_w * 0.002): continue
                
                visual_elements.append({
                    "type": "text",
                    "centroid": (tcx, tcy),
                    "bbox": (bx, by, bw, bh), # 存储bbox
                    "area": tarea * 2.5,
                    "color": (0, 165, 255) # 橙色
                })

            # 计算加权分 & 绘制详细信息
            total_weight = 0
            weighted_score_sum = 0
            
            for item in visual_elements:
                cx_i, cy_i = item['centroid']
                bx, by, bw, bh = item['bbox']
                
                dist_x = min([abs(cx_i - lx) for lx in lines_x])
                dist_y = min([abs(cy_i - ly) for ly in lines_y])
                min_dist = min(dist_x, dist_y)
                
                item_score = max(0, 100 * (1 - min_dist / (diag_len * 0.15)))
                weighted_score_sum += item_score * item['area']
                total_weight += item['area']
                
                # [新增] 绘制边框
                cv2.rectangle(vis_thirds, (bx, by), (bx+bw, by+bh), item['color'], 1)
                
                # 绘制质心
                cv2.circle(vis_thirds, (cx_i, cy_i), 5, item['color'], -1)
                
                # [新增] 绘制单项分数
                cv2.putText(vis_thirds, f"{int(item_score)}", (bx, by-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, item['color'], 1)
                
                # 绘制连接线
                if item_score > 40:
                    if dist_x < dist_y:
                        nx = lines_x[0] if abs(cx_i - lines_x[0]) < abs(cx_i - lines_x[1]) else lines_x[1]
                        cv2.line(vis_thirds, (cx_i, cy_i), (int(nx), cy_i), (0, 255, 0), 1)
                    else:
                        ny = lines_y[0] if abs(cy_i - lines_y[0]) < abs(cy_i - lines_y[1]) else lines_y[1]
                        cv2.line(vis_thirds, (cx_i, cy_i), (cx_i, int(ny)), (0, 255, 0), 1)

            if total_weight > 0:
                score_thirds = weighted_score_sum / total_weight
            else:
                score_thirds = 0.0

            # A3. 平衡
            center_x = process_w // 2
            cv2.line(vis_balance, (center_x, 0), (center_x, new_h), (255, 255, 0), 2)
            m_l = cv2.countNonZero(binary_mask[:, :center_x])
            m_r = cv2.countNonZero(binary_mask[:, center_x:])
            if m_l + m_r > 0:
                score_balance = 100 * (1 - abs(m_l - m_r) / max(m_l, m_r))
                # 绘制力臂 (略微简化绘制)
                ML, MR = cv2.moments(binary_mask[:, :center_x]), cv2.moments(binary_mask[:, center_x:])
                if ML["m00"] > 0:
                    cxl, cyl = int(ML["m10"]/ML["m00"]), int(ML["m01"]/ML["m00"])
                    cv2.circle(vis_balance, (cxl, cyl), 5, (0,255,0), -1)
                    cv2.line(vis_balance, (cxl, cyl), (center_x, cyl), (0,255,0), 2)
                if MR["m00"] > 0:
                    cxr, cyr = int(MR["m10"]/MR["m00"]) + center_x, int(MR["m01"]/MR["m00"])
                    cv2.circle(vis_balance, (cxr, cyr), 5, (0,0,255), -1)
                    cv2.line(vis_balance, (center_x, cyr), (cxr, cyr), (0,0,255), 2)

        # A4. 视觉稳定性
        try:
            k_blur = int(th_sym_blur)
            if k_blur % 2 == 0: k_blur += 1
            img_blurred = cv2.GaussianBlur(img_small, (k_blur, k_blur), 0)
            cx_sym = process_w // 2
            left_half = img_blurred[:, :cx_sym]
            right_half = img_blurred[:, -cx_sym:]
            right_flipped = cv2.flip(right_half, 1)
            
            l_float = left_half.astype(np.float32)
            r_float = right_flipped.astype(np.float32)
            
            diff_map = np.linalg.norm(l_float - r_float, axis=2)
            mean_diff = np.mean(diff_map)
            score_symmetry = max(0, 100 * (1 - mean_diff / th_sym))
            
            diff_full = np.hstack((diff_map, cv2.flip(diff_map, 1)))
            heatmap_norm = cv2.normalize(diff_full, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis_symmetry_heatmap = cv2.cvtColor(cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        except:
            score_symmetry = 0.0
            vis_symmetry_heatmap = None

        # --- 4. 模块 B: 色彩 (Luv) ---
        L, u, v = cv2.split(img_luv)
        L_float = L.astype(np.float32)
        L_norm = L_float / 255.0
        u_float = u.astype(np.float32) - 128.0
        v_float = v.astype(np.float32) - 128.0
        
        chroma = np.sqrt(u_float**2 + v_float**2)
        sat_mean = min(1.0, float(np.mean(chroma) / 128.0))
        chroma_vis = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_saturation = cv2.cvtColor(cv2.applyColorMap(chroma_vis, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        
        bri_mean = float(np.mean(L_norm))
        vis_brightness = cv2.cvtColor(L, cv2.COLOR_GRAY2RGB)
        
        cont_std = float(np.std(L_norm))
        vis_contrast = np.zeros_like(L)
        vis_contrast[L < 85] = 0
        vis_contrast[(L >= 85) & (L < 170)] = 127
        vis_contrast[L >= 170] = 255
        vis_contrast = cv2.cvtColor(vis_contrast, cv2.COLOR_GRAY2RGB)
        
        clarity_mask = (L_norm >= th_clarity)
        clarity_ratio = float(np.count_nonzero(clarity_mask) / L_norm.size)
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
            
            grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(grad_x, grad_y)
            tex_fg = np.mean(mag[binary_mask > 0])
            tex_bg = np.mean(mag[binary_mask_inv > 0])
            texture_diff = min(1.0, abs(tex_fg - tex_bg) / ref_tex)
            mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            edge_fg = cv2.bitwise_and(mag_vis, mag_vis, mask=binary_mask)
            edge_bg = cv2.bitwise_and(mag_vis, mag_vis, mask=binary_mask_inv)
            composite = np.zeros_like(img_small, dtype=np.uint8)
            composite[:, :, 0] = edge_bg
            composite[:, :, 1] = edge_fg
            vis_edge_composite = composite

        # --- 6. 模块 D: 文字分析 ---
        vis_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(vis_pil)
        font = self._load_safe_font(16)
        text_scores = []
        text_contrasts = []
        detected_texts = []
        
        for (bbox, text_content, prob) in ocr_raw:
            if prob > 0.3: detected_texts.append(text_content)
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
                s_con = min(100, contrast)
                item_score = 0.7 * s_con + 0.3 * 80 
                text_scores.append(item_score)
                text_contrasts.append(contrast)
                
                is_legible = item_score > th_text_score
                color = (0, 255, 0) if is_legible else (255, 0, 0)
                draw.rectangle([x, y, x+w_box, y+h_box], outline=color, width=2)
                
                try:
                    label = f"{text_content} | {int(item_score)}"
                    l_box = draw.textbbox((x, y), label, font=font)
                except:
                    label = f"Score: {int(item_score)}"
                    l_box = draw.textbbox((x, y), label, font=font)
                draw.rectangle(l_box, fill=color)
                draw.text((x, y), label, fill=(255, 255, 255), font=font)
            except: continue
                
        has_text = len(text_scores) > 0
        avg_text_score = float(np.mean(text_scores)) if has_text else 0.0
        avg_text_contrast = float(np.mean(text_contrasts)) if has_text else 0.0
        text_content_str = " | ".join(detected_texts) if has_text else "无"
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
            fg_text_content=text_content_str,
            
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

# === 诊断报告生成器 (保持不变) ===
class AestheticDiagnostician:
    @staticmethod
    def generate_report(data: OmniReport, config: Dict = None) -> dict:
        if config is None: config = {}
        
        w_c_diag = config.get('w_comp_diagonal', 1.0)
        w_c_third = config.get('w_comp_thirds', 1.0)
        w_c_bal = config.get('w_comp_balance', 1.0)
        w_c_sym = config.get('w_comp_symmetry', 1.0)
        
        w_l_warm = config.get('w_color_warmth', 0.5)      
        w_l_sat = config.get('w_color_saturation', 0.5)
        w_l_bri = config.get('w_color_brightness', 0.5)   
        w_l_cont = config.get('w_color_contrast', 0.5)
        w_l_clar = config.get('w_color_clarity', 2.0)
        
        w_f_area = config.get('w_fg_area', 1.0)
        w_f_color = config.get('w_fg_color', 1.5)
        w_f_tex = config.get('w_fg_texture', 1.0)
        w_f_text = config.get('w_fg_text', 2.0)

        s_c_diag = data.composition_diagonal
        s_c_third = data.composition_thirds
        s_c_bal = data.composition_balance
        s_c_sym = getattr(data, 'composition_symmetry', 0)

        s_l_warm = data.color_warmth * 100
        s_l_sat = data.color_saturation * 100
        
        val_bri = data.color_brightness
        if 0.45 <= val_bri <= 0.75:
            s_l_bri = 100
        elif val_bri < 0.45:
            s_l_bri = max(0, 100 - (0.45 - val_bri) * 200)
        else:
            s_l_bri = max(0, 100 - (val_bri - 0.75) * 300)

        s_l_cont = min(100, (data.color_contrast / 0.25) * 100)
        
        val_clar = data.color_clarity
        if val_clar > 0.5: 
            s_l_clar = max(0, 100 - (val_clar - 0.5) * 200)
        else:
            s_l_clar = min(100, (val_clar / 0.2) * 100)
            
        s_f_area = data.fg_area_diff * 100
        s_f_color = min(100, data.fg_color_diff)
        s_f_tex = np.sqrt(data.fg_texture_diff) * 100
        
        has_text = getattr(data, 'fg_text_present', False)
        s_f_text = data.fg_text_legibility if has_text else 0
        if not has_text: w_f_text = 0.0

        numerator = (
            s_c_diag * w_c_diag + s_c_third * w_c_third + s_c_bal * w_c_bal + s_c_sym * w_c_sym +
            s_l_warm * w_l_warm + s_l_sat * w_l_sat + s_l_bri * w_l_bri + s_l_cont * w_l_cont + s_l_clar * w_l_clar +
            s_f_area * w_f_area + s_f_color * w_f_color + s_f_tex * w_f_tex + s_f_text * w_f_text
        )
        
        denominator = (
            w_c_diag + w_c_third + w_c_bal + w_c_sym +
            w_l_warm + w_l_sat + w_l_bri + w_l_cont + w_l_clar +
            w_f_area + w_f_color + w_f_tex + w_f_text
        )
        
        total_score = int(numerator / denominator) if denominator > 0 else 0

        if total_score >= 85: rating = "S (大师级)"
        elif total_score >= 70: rating = "A (优秀)"
        elif total_score >= 60: rating = "B (合格)"
        else: rating = "C (需改进)"

        tags = []
        if data.color_warmth > 0.6: tags.append("暖色调")
        elif data.color_warmth < 0.3: tags.append("冷色调")
        if data.color_saturation > 0.5: tags.append("高饱和")
        if data.color_brightness > 0.6: tags.append("高调")
        elif data.color_brightness < 0.3: tags.append("低调")
        if data.composition_diagonal > 80: tags.append("动感构图")
        
        pros, cons, suggestions = [], [], []
        
        if data.composition_symmetry > 90: pros.append("极佳的视觉秩序感")
        if data.color_clarity > 0.15: pros.append("光影通透")
        if data.fg_color_diff > 80: pros.append("主体色彩醒目")
        if data.color_warmth > 0.6: pros.append("色调温暖诱人")
        
        if data.color_clarity < 0.05: 
            cons.append("画面沉闷雾感重")
            suggestions.append("建议提高高光亮度或去雾")
        if data.color_clarity > 0.85:
            cons.append("高光溢出严重")
            suggestions.append("降低曝光防止死白")
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

# === [新增] 标杆管理与相对评分系统 ===
class BenchmarkManager:
    def __init__(self):
        self.metric_keys = [
            'composition_diagonal', 'composition_thirds', 'composition_balance', 'composition_symmetry',
            'color_warmth', 'color_saturation', 'color_brightness', 'color_contrast', 'color_clarity',
            'fg_area_diff', 'fg_color_diff', 'fg_texture_diff',
            'fg_text_legibility' 
        ]

    @staticmethod
    def auto_calculate_tolerance(values: List[float]) -> (float, float):
        mu = float(np.mean(values))
        sigma = float(np.std(values))
        if sigma == 0.0:
            sigma = (abs(mu) * 0.1) if mu != 0.0 else 0.05
        return mu, sigma

    def create_profile(self, reports: List[OmniReport]) -> Dict:
        profile = {}
        count = len(reports)
        if count == 0: return None

        for key in self.metric_keys:
            values = []
            for r in reports:
                val = getattr(r, key, 0)
                if val is None: val = 0
                values.append(float(val))
            mu, sigma = self.auto_calculate_tolerance(values)
            profile[key] = {
                'target': mu,
                'tolerance': sigma
            }
            
        return profile

    def score_against_benchmark(self, data: OmniReport, profile: Dict) -> dict:
        scores = {}
        total_score = 0
        valid_keys = 0
        details = {}

        for key in self.metric_keys:
            if key not in profile:
                continue
            item = profile[key]
            if not isinstance(item, dict) or 'target' not in item or 'tolerance' not in item:
                continue
            target = float(item['target'])
            sigma = float(item['tolerance'])
            actual = getattr(data, key, 0)
            if actual is None:
                actual = 0
            actual = float(actual)
            k = 2.0
            if sigma == 0:
                sigma = 0.05
            if key in ['color_clarity', 'fg_text_legibility', 'fg_color_diff']:
                if actual > target:
                    score = 100.0
                else:
                    score = 100 * np.exp(-((actual - target) ** 2) / (2 * (sigma) ** 2))
            else:
                score = 100 * np.exp(-((actual - target) ** 2) / (2 * (sigma) ** 2))
            scores[key] = score
            total_score += score
            valid_keys += 1
            details[key] = {
                "actual": actual,
                "target": target,
                "score": score
            }

        final_score = int(total_score / valid_keys) if valid_keys > 0 else 0
        if final_score >= 85:
            rating = "S (符合标杆)"
        elif final_score >= 70:
            rating = "A (近似)"
        elif final_score >= 50:
            rating = "B (偏差)"
        else:
            rating = "C (离谱)"
        return {
            "total_score": final_score,
            "rating_level": rating,
            "details": details,
            "profile_used": True
        }