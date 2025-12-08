import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import platform
import os
from scipy.spatial import cKDTree
from scipy.stats import entropy
import math
import base64
import json

# === 引入依赖库 ===
try:
    from rembg import remove, new_session
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    ADVANCED_SEG_AVAILABLE = True
except ImportError:
    ADVANCED_SEG_AVAILABLE = False
    print("⚠️ 未检测到 rembg 或 segment-anything，将回退到基础模式。")

try:
    from openai import OpenAI
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    print("⚠️ 'openai' library not installed. VLM features disabled.")

# PaddleOCR check
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("⚠️ 未检测到 PaddleOCR。建议 'pip install paddlepaddle paddleocr' 以提升中文艺术字识别能力。")

# ==========================================
# 0. 提示词模板
# ==========================================
DEFAULT_ANALYSIS_PROMPT = """你是一位资深的视觉美学总监。请分析{context_str}，并以严格的 JSON 格式返回结果。
JSON 结构如下：
{
    "style": "简短的风格定义 (如: 极简主义/赛博朋克/日系清新/工业风)",
    "score": 0-100之间的整数评分 (基于构图、色彩、光影、质感的专业评估),
    "critique": "一句话的专业点评，指出优点或改进点 (30字以内)"
}
不要输出任何 Markdown 标记，仅输出 JSON 字符串。"""

# === Model Registry ===
class ModelRegistry:
    _instances = {}
    SAM_CHECKPOINT = "sam_vit_b_01ec64.pth" 
    SAM_TYPE = "vit_b"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_u2net_session(cls):
        if 'u2net' not in cls._instances:
            try:
                cls._instances['u2net'] = new_session(model_name="u2net")
            except Exception as e:
                print(f"⚠️ Failed to load U2-Net: {e}")
                return None
        return cls._instances['u2net']

    @classmethod
    def get_sam_predictor(cls):
        if 'sam' not in cls._instances:
            if os.path.exists(cls.SAM_CHECKPOINT) and ADVANCED_SEG_AVAILABLE:
                try:
                    print(f"🔄 Loading SAM Model ({cls.SAM_TYPE})...")
                    sam = sam_model_registry[cls.SAM_TYPE](checkpoint=cls.SAM_CHECKPOINT)
                    sam.to(device=cls.DEVICE)
                    cls._instances['sam'] = SamPredictor(sam)
                except Exception as e:
                    print(f"⚠️ Failed to load SAM: {e}")
                    return None
            else:
                return None
        return cls._instances.get('sam')
    
    @classmethod
    def get_yolo_pose(cls):
        from ultralytics import YOLO
        if 'yolo_pose' not in cls._instances:
            try:
                cls._instances['yolo_pose'] = YOLO("yolov8n-pose.pt")
            except Exception:
                return None
        return cls._instances['yolo_pose']

# === OmniReport Data Structure ===
@dataclass
class OmniReport:
    # --- 构图与视觉秩序 ---
    comp_balance_score: float
    comp_balance_center: Tuple[int, int]
    comp_layout_type: str
    comp_layout_score: float
    comp_negative_space_score: float
    comp_negative_entropy: float
    comp_visual_flow_score: float
    comp_visual_order_score: float
    comp_vanishing_point: Optional[Tuple[int, int]]
    
    # --- 色彩氛围 ---
    color_warmth: float
    color_saturation: float
    color_brightness: float
    color_contrast: float
    color_clarity: float
    color_harmony: float        
    kobayashi_tags: List[str]   
    
    # --- 语义/VLM ---
    semantic_style: str         
    semantic_score: float       
    vlm_critique: str           

    # --- 图底/文字 ---
    fg_area_diff: float
    fg_color_diff: float
    fg_texture_diff: float
    fg_text_present: bool
    fg_text_legibility: float
    fg_text_contrast: float
    fg_text_content: str
    
    # --- 文字排版 ---
    text_alignment_score: float
    text_hierarchy_score: float
    text_content_ratio: float

    # --- 分布 (Legacy) ---
    dist_count: int
    dist_entropy: float         
    dist_cv: float              
    dist_size_cv: float         
    dist_angle_entropy: float   

    # --- 可视化 ---
    vis_mask: Optional[np.ndarray] = None        
    vis_all_elements: Optional[np.ndarray] = None 
    vis_edge_fg: Optional[np.ndarray] = None
    vis_edge_bg: Optional[np.ndarray] = None
    vis_edge_composite: Optional[np.ndarray] = None
    vis_text_analysis: Optional[np.ndarray] = None
    vis_text_design: Optional[np.ndarray] = None 
    
    vis_color_contrast: Optional[np.ndarray] = None
    vis_symmetry_heatmap: Optional[np.ndarray] = None
    
    vis_saliency_heatmap: Optional[np.ndarray] = None
    vis_layout_template: Optional[np.ndarray] = None # Legacy
    vis_layout_dict: Optional[Dict] = None # New dict
    vis_visual_flow: Optional[np.ndarray] = None
    vis_visual_order: Optional[np.ndarray] = None
    
    vis_diag: Optional[np.ndarray] = None
    vis_thirds: Optional[np.ndarray] = None
    vis_balance: Optional[np.ndarray] = None
    vis_clarity: Optional[np.ndarray] = None
    vis_warmth: Optional[np.ndarray] = None
    vis_saturation: Optional[np.ndarray] = None
    vis_brightness: Optional[np.ndarray] = None
    vis_contrast: Optional[np.ndarray] = None
    vis_color_harmony: Optional[np.ndarray] = None 
    vis_dist_entropy: Optional[np.ndarray] = None 
    vis_dist_size: Optional[np.ndarray] = None     
    vis_dist_angle: Optional[np.ndarray] = None
    
    # Legacy placeholders with defaults
    composition_diagonal: float = 0.0
    composition_thirds: float = 0.0
    composition_balance: float = 0.0
    composition_symmetry: float = 0.0

# === 豆包/VLM 分析器 ===
class DoubaoVLMAnalyzer:
    def __init__(self, api_key=None, model_endpoint=None):
        self.client = None
        self.model_endpoint = model_endpoint
        if VLM_AVAILABLE and api_key and model_endpoint:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://ark-cn-beijing.bytedance.net/api/v3",
            )

    def _encode_image(self, image_bgr):
        _, buffer = cv2.imencode('.jpg', image_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze(self, image_bgr, subject_mask=None, custom_prompt_template=None):
        if not self.client:
            return "未配置API", 0.0, "请在侧边栏配置豆包 API Key 和 Endpoint ID"

        target_img = image_bgr
        if subject_mask is not None and cv2.countNonZero(subject_mask) > 0:
            x, y, w, h = cv2.boundingRect(subject_mask)
            if w > 10 and h > 10:
                pad = 10
                h_img, w_img = image_bgr.shape[:2]
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(w_img, x + w + pad); y2 = min(h_img, y + h + pad)
                target_img = image_bgr[y1:y2, x1:x2]

        base64_image = self._encode_image(target_img)
        
        template = custom_prompt_template if custom_prompt_template and custom_prompt_template.strip() else DEFAULT_ANALYSIS_PROMPT
        try:
            system_prompt = template.format(context_str="这张图片")
        except Exception:
            system_prompt = template

        try:
            response = self.client.chat.completions.create(
                model=self.model_endpoint,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "请进行专业美学分析。"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}],
                    },
                ],
            )
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            return result.get("style", "未知"), float(result.get("score", 0)), result.get("critique", "解析失败")
        except Exception as e:
            print(f"VLM Analysis Error: {e}")
            return "API错误", 0.0, f"调用失败: {str(e)}"

# === CIECAM02 向量化实现 ===
class CIECAM02_Vectorized:
    def __init__(self):
        self.M_CAT02 = np.array([[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.0030, 0.0136, 0.9834]])
        self.M_HPE = np.array([[0.38971, 0.68898, -0.07868], [-0.22981, 1.18340, 0.04641], [0.00000, 0.00000, 1.00000]])
        self.La = 64.0; self.Yb = 20.0; self.Yw = 100.0
        self.xw = 0.95047; self.yw = 1.00000; self.zw = 1.08883
        self.F = 1.0; self.c = 0.69; self.Nc = 1.0
        k = 1 / (5 * self.La + 1); self.k4 = k ** 4
        self.FL = 0.2 * k ** 4 * (5 * self.La) + 0.1 * (1 - self.k4)**2 * (5 * self.La)**(1/3)
        self.n = self.Yb / self.Yw; self.z = 1.48 + np.sqrt(self.n)
        self.Nbb = 0.725 * (1/self.n)**0.2; self.Cbw = 1.0
        self.RGB_w = np.dot(self.M_CAT02, np.array([self.xw*100, self.yw*100, self.zw*100]))
        self.D = np.clip(self.F * (1 - (1/3.6) * np.exp((-self.La-42)/92)), 0, 1)
        self.Rc_factor = ((self.Yw * self.D) / self.RGB_w[0]) + (1 - self.D)
        self.Gc_factor = ((self.Yw * self.D) / self.RGB_w[1]) + (1 - self.D)
        self.Bc_factor = ((self.Yw * self.D) / self.RGB_w[2]) + (1 - self.D)
        RGB_cw = np.array([self.RGB_w[0]*self.Rc_factor, self.RGB_w[1]*self.Gc_factor, self.RGB_w[2]*self.Bc_factor])
        RGB_prime_w = np.dot(np.dot(self.M_HPE, np.linalg.inv(self.M_CAT02)), RGB_cw)
        RGB_aw_prime = 400 * (self.FL * RGB_prime_w / 100)**0.42 / (27.13 + (self.FL * RGB_prime_w / 100)**0.42) + 0.1
        self.Aw = 2.0 * RGB_aw_prime[0] + 1.0 * RGB_aw_prime[1] + 0.05 * RGB_aw_prime[2] - 0.305

    def forward(self, img_xyz):
        h, w, c = img_xyz.shape; XYZ_flat = img_xyz.reshape(-1, 3).T
        RGB = np.dot(self.M_CAT02, XYZ_flat); RGB_c = np.empty_like(RGB)
        RGB_c[0] = RGB[0] * self.Rc_factor; RGB_c[1] = RGB[1] * self.Gc_factor; RGB_c[2] = RGB[2] * self.Bc_factor
        M_combined = np.dot(self.M_HPE, np.linalg.inv(self.M_CAT02)); RGB_prime = np.dot(M_combined, RGB_c)
        RGB_prime_abs = np.abs(RGB_prime); sign = np.sign(RGB_prime)
        temp = (self.FL * RGB_prime_abs / 100.0) ** 0.42
        RGB_a_prime = sign * 400 * temp / (27.13 + temp) + 0.1
        Ra, Ga, Ba = RGB_a_prime[0], RGB_a_prime[1], RGB_a_prime[2]
        a = Ra - 12 * Ga / 11 + Ba / 11; b = (1/9) * (Ra + Ga - 2 * Ba)
        h_rad = np.arctan2(b, a); h_deg = np.degrees(h_rad) % 360.0
        et = 0.25 * (np.cos(h_rad + 2) + 3.8); A = (2.0 * Ra + 1.0 * Ga + 0.05 * Ba - 0.305) * self.Nbb
        J = 100 * (np.maximum(0, A) / self.Aw) ** (self.c * self.z)
        t = (50000 / 13) * self.Nc * self.Nbb * et * np.sqrt(a**2 + b**2) / (Ra + Ga + 21 * Ba / 20)
        C = t**0.9 * np.sqrt(J / 100) * (1.64 - 0.29**self.n)**0.73; M = C * self.FL**0.25
        return {'J': J.reshape(h, w), 'C': C.reshape(h, w), 'h': h_deg.reshape(h, w), 'M': M.reshape(h, w)}

# === 混合分割器 ===
class HybridSegmenter:
    def __init__(self):
        self.u2net_session = ModelRegistry.get_u2net_session()
        self.sam_predictor = ModelRegistry.get_sam_predictor()
        if not self.sam_predictor:
            print("⚠️ SAM 未加载，将仅使用 U2-Net 进行粗略分割。")

    def extract_main_subject_mask(self, image_bgr: np.ndarray, config: Dict = None, text_boxes: List = None) -> Tuple[np.ndarray, List]:
        h, w = image_bgr.shape[:2]
        box_prompts = []
        debug_boxes = []
        
        final_mask = np.zeros((h, w), dtype=np.uint8)
        EXPAND_RATIO = 0.05
        
        def add_box_with_padding(x1, y1, x2, y2, label_type):
            bw = x2 - x1; bh = y2 - y1
            pad_x = int(bw * EXPAND_RATIO); pad_y = int(bh * EXPAND_RATIO)
            nx1 = max(0, x1 - pad_x); ny1 = max(0, y1 - pad_y)
            nx2 = min(w, x2 + pad_x); ny2 = min(h, y2 + pad_y)
            if nx2 > nx1 and ny2 > ny1:
                box_prompts.append([nx1, ny1, nx2, ny2])
                debug_boxes.append({'box': [nx1, ny1, nx2-nx1, ny2-nx1], 'type': label_type})

        # 1. U2-Net
        if self.u2net_session:
            try:
                img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                saliency_mask_pil = remove(Image.fromarray(img_rgb), session=self.u2net_session, only_mask=True)
                saliency_mask = np.array(saliency_mask_pil)
                _, thresh = cv2.threshold(saliency_mask, 100, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_area = (h * w) * 0.001
                u2net_valid_mask = np.zeros((h, w), dtype=np.uint8)
                for cnt in contours:
                    if cv2.contourArea(cnt) > min_area:
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        add_box_with_padding(x, y, x + bw, y + bh, 'object')
                        cv2.drawContours(u2net_valid_mask, [cnt], -1, 255, -1)
                final_mask = cv2.bitwise_or(final_mask, u2net_valid_mask)
            except Exception as e:
                print(f"U2-Net Error: {e}")

        # Fallback
        if not box_prompts and cv2.countNonZero(final_mask) == 0:
             saliency_mask = self._fallback_saliency(image_bgr)
             _, thresh = cv2.threshold(saliency_mask, 100, 255, cv2.THRESH_BINARY)
             contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
             for cnt in contours:
                 x, y, bw, bh = cv2.boundingRect(cnt)
                 add_box_with_padding(x, y, x + bw, y + bh, 'fallback')
                 cv2.drawContours(final_mask, [cnt], -1, 255, -1)

        # 2. OCR Text Boxes
        if text_boxes:
            for bbox in text_boxes:
                pts = np.array(bbox, dtype=np.int32)
                x, y, bw, bh = cv2.boundingRect(pts)
                if bw > 2 and bh > 2:
                    add_box_with_padding(x, y, x + bw, y + bh, 'text')

        # 3. SAM
        if self.sam_predictor and box_prompts:
            try:
                self.sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                for box in box_prompts:
                    input_box = np.array(box)
                    masks, _, _ = self.sam_predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False)
                    mask_uint8 = (masks[0] * 255).astype(np.uint8)
                    final_mask = cv2.bitwise_or(final_mask, mask_uint8)
            except Exception as e:
                print(f"SAM Inference Error: {e}")
        
        # 4. Box Coverage Guarantee
        for box in box_prompts:
            x1, y1, x2, y2 = box
            mask_roi = final_mask[y1:y2, x1:x2]
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > 0:
                fill_ratio = cv2.countNonZero(mask_roi) / box_area
                if fill_ratio < 0.15:
                    cv2.rectangle(final_mask, (x1, y1), (x2, y2), 255, -1)

        # 5. [Force] 强制叠加文字区域
        if text_boxes:
            for bbox in text_boxes:
                pts = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(final_mask, [pts], 255)
            
        return final_mask, debug_boxes

    def _fallback_saliency(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return mask

# === 全能视觉分析引擎 ===
class OmniVisualEngine:
    def __init__(self, vlm_api_key=None, vlm_endpoint=None):
        print("Initializing Omni Engine v18.4 (Final Fix)...")
        self.ocr_type = 'easyocr'
        self.ocr_reader = None
        
        if PADDLE_AVAILABLE:
            try:
                print("🔄 Loading PaddleOCR...")
                self.ocr_reader = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
                self.ocr_type = 'paddle'
            except Exception as e:
                print(f"⚠️ PaddleOCR Init Failed: {e}")
        
        if self.ocr_reader is None:
            print("🔄 Loading EasyOCR (Fallback)...")
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
            self.ocr_type = 'easyocr'

        self.segmenter = HybridSegmenter()
        self.pose_model = ModelRegistry.get_yolo_pose()
        self.cam02 = CIECAM02_Vectorized()
        self.vlm_analyzer = DoubaoVLMAnalyzer(api_key=vlm_api_key, model_endpoint=vlm_endpoint)

    def _load_safe_font(self, font_size=16):
        return ImageFont.load_default()

    def _run_ocr(self, img):
        if self.ocr_type == 'paddle':
            try:
                result = self.ocr_reader.ocr(img, cls=True)
                output = []
                if result and result[0]:
                    for line in result[0]:
                        points = np.array(line[0], dtype=np.int32)
                        text = line[1][0]
                        conf = line[1][1]
                        output.append((points, text, conf))
                return output
            except Exception as e:
                print(f"PaddleOCR Error: {e}")
                return []
        else:
            raw = self.ocr_reader.readtext(img)
            output = []
            for item in raw:
                points = np.array(item[0], dtype=np.int32)
                output.append((points, item[1], item[2]))
            return output

    # --- Helpers ---
    def _draw_dist_entropy_map(self, vis, grid_counts, w, h):
        grid_size = grid_counts.shape[0]; step_x = w/grid_size; step_y = h/grid_size
        max_c = np.max(grid_counts) if np.max(grid_counts) > 0 else 1
        overlay = vis.copy()
        for y in range(grid_size):
            for x in range(grid_size):
                count = grid_counts[y,x]
                if count > 0:
                    it = count/max_c; col = (int(255*(1-it)), 100, int(255*it)) 
                    cv2.rectangle(overlay, (int(x*step_x),int(y*step_y)), (int((x+1)*step_x),int((x+1)*step_y)), col, -1)
                    cv2.putText(overlay, str(int(count)), (int(x*step_x)+5, int(y*step_y)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)

    def _draw_dist_size_map(self, vis, visual_elements, valid_contours):
        vis = cv2.addWeighted(vis, 0.3, np.zeros_like(vis), 0.7, 0)
        if not visual_elements: return vis
        areas = [x['area'] for x in visual_elements]; max_area = max(areas) if areas else 1
        for item, cnt in zip(visual_elements, valid_contours):
            norm_size = item['area'] / max_area
            b = int(255 * (1 - norm_size)); g = int(100 * (1 - abs(norm_size - 0.5)*2)); r = int(255 * norm_size)
            cv2.drawContours(vis, [cnt], -1, (b, g, r), -1); cv2.drawContours(vis, [cnt], -1, (255,255,255), 1)
            cx, cy = item['centroid']; cv2.putText(vis, f"S:{int(item['area'])}", (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        return vis

    def _draw_dist_angle_map(self, vis, visual_elements, valid_contours):
        vis = cv2.addWeighted(vis, 0.3, np.zeros_like(vis), 0.7, 0)
        for item, cnt in zip(visual_elements, valid_contours):
            cx, cy = item['centroid']; angle = item['angle']
            cv2.drawContours(vis, [cnt], -1, (100,100,100), 1)
            length = 40; rad = math.radians(angle)
            end_x = int(cx + length * math.cos(rad)); end_y = int(cy + length * math.sin(rad))
            cv2.line(vis, (cx, cy), (end_x, end_y), (0, 255, 255), 2); cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(vis, f"{int(angle)}d", (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        return vis

    def _calc_perceptual_balance(self, img, saliency_mask):
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        if saliency_mask is None: return 0.0, (center_x, center_y), None
        moments = cv2.moments(saliency_mask)
        if moments["m00"] == 0: return 0.0, (center_x, center_y), None
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        dist = math.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        max_dist = math.sqrt(center_x**2 + center_y**2)
        offset_ratio = min(1.0, dist / (max_dist * 0.5))
        balance_score = max(0, 100 * (1 - offset_ratio))
        heatmap = cv2.applyColorMap(saliency_mask, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        cv2.circle(vis, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)
        cv2.line(vis, (center_x, center_y), (cx, cy), (0, 255, 255), 2)
        return balance_score, (cx, cy), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    def _match_composition_template(self, mask, w, h, img_bg):
        best_iou = 0.0
        best_type = "Unknown"
        templates = {}
        
        center = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(center, (w//2, h//2), (w//3, h//3), 0, 0, 360, 255, -1)
        templates["Center"] = center
        
        thirds = np.zeros((h, w), dtype=np.uint8)
        dw, dh = w//3, h//3
        r = min(w, h) // 6
        points = [(dw, dh), (2*dw, dh), (dw, 2*dh), (2*dw, 2*dh)]
        for pt in points: cv2.circle(thirds, pt, r, 255, -1)
        templates["Rule of Thirds"] = thirds
        
        diag = np.zeros((h, w), dtype=np.uint8)
        thickness = min(w, h) // 4
        cv2.line(diag, (0, h), (w, 0), 255, thickness)
        cv2.line(diag, (0, 0), (w, h), 255, thickness)
        templates["Diagonal"] = diag
        
        tri = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([[w//2, h//4], [w//6, h], [5*w//6, h]], np.int32)
        cv2.fillPoly(tri, [pts], 255)
        templates["Triangle"] = tri

        frame = np.zeros((h, w), dtype=np.uint8)
        border_thick = min(w, h) // 6
        cv2.rectangle(frame, (0, 0), (w, h), 255, border_thick)
        templates["Frame"] = frame

        s_curve = np.zeros((h, w), dtype=np.uint8)
        s_pts = []
        for y in range(0, h, 10):
            angle = (y / h) * 2 * math.pi
            x = int(w/2 + (w/3) * math.sin(angle)) 
            s_pts.append([x, y])
        if s_pts:
            cv2.polylines(s_curve, [np.array(s_pts, np.int32)], False, 255, min(w, h) // 5)
        templates["S-Curve"] = s_curve

        mask_bool = mask > 127
        results = {}
        for name, temp in templates.items():
            temp_bool = temp > 0
            intersection = np.logical_and(mask_bool, temp_bool).sum()
            union = np.logical_or(mask_bool, temp_bool).sum()
            iou = intersection / union if union > 0 else 0
            score = min(100, iou * 100 * 1.5)
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            vis[mask_bool] = (0, 0, 255)
            vis[temp_bool] = (0, 255, 0)
            vis[np.logical_and(mask_bool, temp_bool)] = (0, 255, 255) 
            vis = cv2.addWeighted(img_bg, 0.5, vis, 0.5, 0)
            cv2.putText(vis, f"{name}: {int(score)}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            results[name] = {'score': score, 'vis': cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)}
            if iou > best_iou:
                best_iou = iou
                best_type = name
        score = min(100, best_iou * 100 * 1.5)
        return best_type, score, results

    def _analyze_negative_space(self, bg_mask):
        h, w = bg_mask.shape[:2]
        total_pixels = h * w
        bg_pixels = cv2.countNonZero(bg_mask)
        if bg_pixels == 0: return 0.0, 0.0
        grid_h, grid_w = 10, 10
        step_h, step_w = h // grid_h, w // grid_w
        counts = []
        for i in range(grid_h):
            for j in range(grid_w):
                roi = bg_mask[i*step_h:(i+1)*step_h, j*step_w:(j+1)*step_w]
                c = cv2.countNonZero(roi)
                if c > 0: counts.append(c)
        probs = np.array(counts) / bg_pixels
        bg_entropy = entropy(probs)
        max_ent = np.log(len(counts)) if len(counts) > 0 else 1
        norm_entropy = bg_entropy / max_ent if max_ent > 0 else 0
        bg_ratio = bg_pixels / total_pixels
        ratio_score = 100 - abs(bg_ratio - 0.55) * 200
        breath_score = max(0, min(100, ratio_score * (1 - norm_entropy*0.5)))
        return breath_score, norm_entropy

    def _analyze_visual_flow(self, img_gray, mask):
        h, w = img_gray.shape
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(img_gray)[0]
        vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        if mask is not None:
            mask_indices = mask > 0
            if np.any(mask_indices):
                overlay = vis.copy()
                overlay[mask_indices] = (0, 0, 255)
                vis = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)
        valid_intersections = []
        if lines is not None:
            long_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > min(h, w) * 0.1:
                    long_lines.append(line[0])
                    cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
            if len(long_lines) > 2:
                for i in range(min(50, len(long_lines))):
                    for j in range(i+1, min(50, len(long_lines))):
                        l1 = long_lines[i]
                        l2 = long_lines[j]
                        x1, y1, x2, y2 = l1
                        x3, y3, x4, y4 = l2
                        denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
                        if denom == 0: continue
                        ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
                        ix = x1 + ua * (x2-x1); iy = y1 + ua * (y2-y1)
                        if 0 <= ix < w and 0 <= iy < h: valid_intersections.append((int(ix), int(iy)))
        vp = None; score = 0.0
        if valid_intersections:
            grid_sz = 50
            grid = np.zeros((h//grid_sz + 1, w//grid_sz + 1))
            for ix, iy in valid_intersections: grid[iy//grid_sz, ix//grid_sz] += 1
            my, mx = np.unravel_index(grid.argmax(), grid.shape)
            vp = (mx * grid_sz + grid_sz//2, my * grid_sz + grid_sz//2)
            cv2.circle(vis, vp, 10, (0, 255, 0), -1)
            in_mask = mask[min(h-1, vp[1]), min(w-1, vp[0])] > 0
            score = 90.0 if in_mask else 40.0
        return score, vp, vis

    def _extract_visual_elements(self, image_bgr, binary_mask, ocr_raw, face_points, new_h, process_w):
        visual_elements = []; valid_contours = []
        min_graphic_area = max(10, (new_h * process_w) * 0.005); min_text_area = max(5, (new_h * process_w) * 0.002)
        sub_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_polys = []
        for item in ocr_raw:
            if len(item) >= 3 and item[2] > 0.01: 
                text_polys.append(np.array(item[0], dtype=np.int32))

        for cnt in sub_contours:
            area = cv2.contourArea(cnt)
            if area < min_graphic_area: continue
            M = cv2.moments(cnt)
            if M["m00"] <= 1e-5: continue
            cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
            is_text_overlap = False
            for tp in text_polys:
                if cv2.pointPolygonTest(tp, (cx, cy), False) >= 0:
                    is_text_overlap = True
                    break
            if is_text_overlap: continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            mu20 = M['mu20']; mu02 = M['mu02']; mu11 = M['mu11']
            theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02); angle_deg = math.degrees(theta)
            if angle_deg < 0: angle_deg += 180
            is_face = False
            for fx, fy in face_points:
                if cv2.pointPolygonTest(cnt, (fx, fy), True) > -20: cx, cy = fx, fy; is_face = True; break
            visual_elements.append({
                "type": "graphic", "centroid": (cx, cy), "bbox": (bx, by, bw, bh), "area": area * 2.0 if is_face else area, 
                "angle": angle_deg, "color": (255, 0, 255) if is_face else (0, 0, 255), "contour": cnt 
            }); valid_contours.append(cnt)
            
        for item in ocr_raw:
            if len(item) >= 3 and item[2] > 0.01:
                bbox, text, prob = item[0], item[1], item[2]
                pts = np.array(bbox, dtype=np.int32); bx, by, bw, bh = cv2.boundingRect(pts)
                M_txt = cv2.moments(pts)
                if M_txt["m00"] <= 1e-5: continue
                tcx = int(M_txt["m10"] / M_txt["m00"]); tcy = int(M_txt["m01"] / M_txt["m00"])
                if cv2.contourArea(pts) < min_text_area: continue
                rect = cv2.minAreaRect(pts); t_angle = rect[2]; 
                if rect[1][0] < rect[1][1]: t_angle += 90
                text_cnt = np.array([[[bx, by]], [[bx+bw, by]], [[bx+bw, by+bh]], [[bx, by+bh]]], dtype=np.int32)
                visual_elements.append({
                    "type": "text", "centroid": (tcx, tcy), "bbox": (bx, by, bw, bh), "area": cv2.contourArea(pts) * 2.5, 
                    "angle": abs(t_angle) % 180, "color": (0, 165, 255), "contour": text_cnt
                }); valid_contours.append(text_cnt)
        return visual_elements, valid_contours

    def _analyze_distribution(self, img_bgr, visual_elements, valid_contours):
        h, w = img_bgr.shape[:2]; centroids = []; areas = []; angles = [] 
        for item in visual_elements:
            centroids.append(list(item['centroid'])); areas.append(item['area']); angles.append(item['angle']) 
        num_objects = len(centroids); grid_size = 10; grid_counts = np.zeros((grid_size, grid_size))
        norm_entropy = 0.0
        if num_objects > 0:
            for x, y in centroids:
                grid_counts[min(int(y/h*grid_size), grid_size-1), min(int(x/w*grid_size), grid_size-1)] += 1
            prob = grid_counts.flatten() / num_objects; prob_filtered = prob[prob > 0]
            max_entropy = np.log(grid_size**2); norm_entropy = entropy(prob_filtered) / max_entropy if max_entropy > 0 else 0.0
        spacing_cv = 0.0
        if num_objects >= 2:
            try:
                points = np.array(centroids); tree = cKDTree(points); dists, _ = tree.query(points, k=2)
                nearest = dists[:, 1]; mean_dist = np.mean(nearest)
                spacing_cv = (np.std(nearest) / mean_dist) if mean_dist > 1e-5 else 0.0
            except Exception: pass
        size_cv = 0.0
        if num_objects >= 2 and sum(areas) > 0:
            mean_area = np.mean(areas)
            if mean_area > 1e-5: size_cv = np.std(areas) / mean_area
        angle_entropy = 0.0
        if num_objects >= 2:
            hist_angle, _ = np.histogram(angles, bins=18, range=(0, 180), density=True); hist_angle += 1e-10
            angle_entropy = entropy(hist_angle) / np.log(18) if np.log(18) > 0 else 0.0
        
        visual_order_score = max(0, 100 * (1 - angle_entropy))
        vis_visual_order = self._draw_dist_angle_map(img_bgr.copy(), visual_elements, valid_contours)
        return num_objects, visual_order_score, cv2.cvtColor(vis_visual_order, cv2.COLOR_BGR2RGB)

    def _analyze_text_layout(self, valid_ocr_items, w, h):
        if not valid_ocr_items:
            return 0.0, 0.0, 0.0, None
        
        boxes = [item[0] for item in valid_ocr_items]
        total_text_area = 0
        heights = []
        centers_x = []
        lefts_x = []
        rights_x = []
        
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        for box in boxes:
            area = cv2.contourArea(box)
            total_text_area += area
            x, y, bw, bh = cv2.boundingRect(box)
            heights.append(bh)
            centers_x.append(x + bw/2)
            lefts_x.append(x)
            rights_x.append(x + bw)
            cv2.polylines(vis, [box], True, (100, 100, 100), 1)

        text_content_ratio = min(1.0, total_text_area / (w * h)) * 100
        
        if len(boxes) >= 2:
            std_left = np.std(lefts_x)
            std_center = np.std(centers_x)
            std_right = np.std(rights_x)
            min_std = min(std_left, std_center, std_right)
            norm_std = min_std / w
            alignment_score = max(0, 100 * math.exp(-10 * norm_std))
            
            if min_std == std_left:
                line_x = int(np.mean(lefts_x))
                cv2.line(vis, (line_x, 0), (line_x, h), (0, 255, 0), 2)
            elif min_std == std_center:
                line_x = int(np.mean(centers_x))
                cv2.line(vis, (line_x, 0), (line_x, h), (0, 255, 0), 2)
            else:
                line_x = int(np.mean(rights_x))
                cv2.line(vis, (line_x, 0), (line_x, h), (0, 255, 0), 2)
        else:
            alignment_score = 100.0
            
        hierarchy_score = 0.0
        if len(heights) >= 2:
            max_h = max(heights)
            avg_h = np.mean(heights)
            for i, box in enumerate(boxes):
                h_i = heights[i]
                ratio = h_i / max_h
                color = (0, int(255 * (1-ratio)), int(255 * ratio))
                cv2.fillPoly(vis, [box], color)
            cv_h = np.std(heights) / (avg_h + 1e-5)
            hierarchy_score = min(100, cv_h * 200)
            
        return alignment_score, hierarchy_score, text_content_ratio, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    def _analyze_kobayashi_image_scale(self, h_map, C_map, J_map, M_map):
        mask = (C_map > 5) & (J_map > 10) & (J_map < 98)
        if np.count_nonzero(mask) < 1000: return ["中性 (Neutral)"]
        h_valid = h_map[mask]; C_valid = C_map[mask]; J_valid = J_map[mask]
        wc_score = np.cos(np.radians(h_valid - 40)); weighted_wc = np.average(wc_score, weights=C_valid) if np.sum(C_valid) > 0 else 0
        sh_score = ((J_valid - 50) / 60.0) - (C_valid / 120.0); weighted_sh = np.mean(sh_score)
        tags = []; x, y = weighted_wc, weighted_sh
        if abs(x) < 0.15 and abs(y) < 0.15: tags.extend(["自然", "温和"])
        else:
            if y > 0.2: tags.append("浪漫" if x > 0.2 else "优雅" if x < -0.2 else "轻柔")
            elif y < -0.2: tags.append("动感" if x > 0.2 else "现代" if x < -0.2 else "庄重")
            else: tags.append("休闲" if x > 0.3 else "时尚" if x < -0.3 else "经典")
        if J_valid.mean() > 85: tags.insert(0, "清爽")
        if C_valid.mean() > 60: tags.insert(0, "华丽")
        return list(set(tags))[:3]

    def _analyze_color_harmony(self, h_map, C_map, J_map, M_map, img_bgr):
        h_flat = h_map.flatten(); C_flat = C_map.flatten(); bgr_flat = img_bgr.reshape(-1, 3)
        mask = (C_flat > 5) & (J_map.flatten() > 10) & (J_map.flatten() < 98)
        if np.count_nonzero(mask) < 1000: return 0.0, None 
        h_valid = h_flat[mask]; C_valid = C_flat[mask]; bgr_valid = bgr_flat[mask]
        rads = np.radians(h_valid); a_vals = C_valid * np.cos(rads); b_vals = C_valid * np.sin(rads)
        data = np.vstack((a_vals, b_vals)).T.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5; K = data.shape[0] if data.shape[0] < K else K
        _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten()); total_counts = np.sum(counts); weights = counts / total_counts
        dom_hues = []; dom_colors_bgr = []
        labels_flat = labels.flatten()
        for i in range(K):
            ca, cb = centers[i]; deg = math.degrees(math.atan2(cb, ca)); dom_hues.append(deg if deg >= 0 else deg + 360)
            cluster_pixels = bgr_valid[labels_flat == i]
            if len(cluster_pixels) > 0: dom_colors_bgr.append(tuple(map(int, np.mean(cluster_pixels, axis=0))))
            else: dom_colors_bgr.append((128, 128, 128))
        templates = {"Analogous": [0], "Complementary": [0, 180], "Split": [0, 150, 210], "Triadic": [0, 120, 240]}
        best_score = 0.0; best_type = "None"; best_rotation = 0.0
        for t_name, t_angles in templates.items():
            min_dist_sum = float('inf'); best_rot = 0
            for rot in range(0, 360, 5):
                target = [(a + rot) % 360 for a in t_angles]; curr_dist = 0.0
                for dh, w in zip(dom_hues, weights):
                    curr_dist += min([min(abs(dh - ta), 360 - abs(dh - ta)) for ta in target]) * w
                if curr_dist < min_dist_sum: min_dist_sum = curr_dist; best_rot = rot
            score = max(0, 100 * (1 - min_dist_sum / 30.0))
            if score > best_score: best_score = score; best_type = t_name; best_rotation = best_rot
        vis_size = 400; vis = np.ones((vis_size, vis_size, 3), dtype=np.uint8) * 255; center = (vis_size//2, vis_size//2); radius = vis_size // 2 - 20
        for i in range(360):
            rad = math.radians(i); p_out = (int(center[0] + radius * math.cos(rad)), int(center[1] + radius * math.sin(rad)))
            p_in = (int(center[0] + (radius-20) * math.cos(rad)), int(center[1] + (radius-20) * math.sin(rad)))
            col = cv2.cvtColor(np.array([[[i/2, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
            cv2.line(vis, p_in, p_out, (int(col[0]), int(col[1]), int(col[2])), 2)
        for dh, w, real_col in zip(dom_hues, weights, dom_colors_bgr):
            rad = math.radians(dh); dist = radius - 60
            px = int(center[0] + dist * math.cos(rad)); py = int(center[1] + dist * math.sin(rad))
            r_size = int(10 + w * 40); cv2.circle(vis, (px, py), r_size, real_col, -1); cv2.circle(vis, (px, py), r_size, (0,0,0), 1)
        if best_type != "None":
            for a in templates[best_type]:
                rad = math.radians((a + best_rotation) % 360)
                cv2.line(vis, center, (int(center[0] + (radius+10) * math.cos(rad)), int(center[1] + (radius+10) * math.sin(rad))), (50, 50, 50), 2)
            cv2.putText(vis, f"{best_type} ({int(best_score)})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        return best_score, vis

    def analyze(self, image_input: np.ndarray, config: Dict = None) -> 'OmniReport':
        if config is None: config = {}
        process_w = config.get('process_width', 512)
        custom_analysis_prompt = config.get('analysis_prompt')

        # Variables Init
        vis_diag = image_input.copy(); vis_thirds = image_input.copy(); vis_balance = image_input.copy()
        score_diag = 0.0; score_thirds = 0.0; score_balance = 0.0
        score_symmetry = 0.0; vis_symmetry_heatmap = None

        comp_balance_score = 0.0
        comp_balance_center = (0, 0)
        comp_layout_type = "Unknown"
        comp_layout_score = 0.0
        comp_negative_space_score = 0.0
        comp_negative_entropy = 0.0
        comp_visual_flow_score = 0.0
        comp_vanishing_point = None
        vis_saliency_heatmap = image_input.copy()
        vis_layout_dict = None
        vis_layout_template = image_input.copy()
        vis_visual_flow = image_input.copy()
        vis_visual_order = image_input.copy() # Init
        visual_order_score = 0.0 # Init
        
        # New text vars
        text_alignment_score = 0.0
        text_hierarchy_score = 0.0
        text_content_ratio = 0.0
        vis_text_design = None
        
        # Initialize PIL visualizer
        vis_pil = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        
        # Init color/texture vars
        warmth_ratio = 0.0; sat_mean = 0.0; bri_mean = 0.0
        cont_std = 0.0; clarity_ratio = 0.0; harmony_score = 0.0
        area_diff = 0.0; color_diff = 0.0; texture_diff = 0.0
        avg_text_score = 0.0; avg_text_contrast = 0.0; text_content_str = "None"
        
        vis_edge_composite = None; vis_color_contrast = None; vis_text_final = None
        vis_warmth = None; vis_saturation = None; vis_brightness = None; vis_contrast = None; vis_clarity = None; vis_color_harmony = None

        # Init old distribution variables to satisfy OmniReport dataclass, even if they are 0
        d_count = 0; d_ent = 0.0; d_cv = 0.0; d_size_cv = 0.0; d_angle_ent = 0.0
        vis_dist_entropy = None; vis_dist_size = None; vis_dist_angle = None

        h, w = image_input.shape[:2]; scale = process_w / w; new_h = int(h * scale)
        img_small = cv2.resize(image_input, (process_w, new_h))
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB); img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        img_xyz = cv2.cvtColor(img_small, cv2.COLOR_BGR2XYZ).astype(np.float32); img_xyz_norm = img_xyz / 255.0 * 100.0
        cam_res = self.cam02.forward(img_xyz_norm); J, C, h_ang, M = cam_res['J'], cam_res['C'], cam_res['h'], cam_res['M']
        harmony_score, vis_harmony = self._analyze_color_harmony(h_ang, C, J, M, img_small)
        kobayashi_tags = self._analyze_kobayashi_image_scale(h_ang, C, J, M)
        
        ocr_raw = self._run_ocr(img_small)
        valid_ocr_items = [item for item in ocr_raw if item[2] > 0.01]
        text_boxes_low_conf = [item[0] for item in valid_ocr_items]
        
        binary_mask, debug_boxes = self.segmenter.extract_main_subject_mask(
            img_small, config, text_boxes=text_boxes_low_conf
        )
        
        vis_mask_debug = img_small.copy()
        mask_indices = binary_mask > 0
        if np.any(mask_indices):
            overlay = vis_mask_debug.copy()
            overlay[mask_indices] = (0, 0, 255) 
            vis_mask_debug = cv2.addWeighted(vis_mask_debug, 0.7, overlay, 0.3, 0)
        
        for item in debug_boxes:
            box = item['box']
            x, y, bw, bh = box
            color = (0, 255, 0) if item['type'] == 'object' else (255, 0, 0) # Green for Object, Blue for Text
            cv2.rectangle(vis_mask_debug, (x, y), (x + bw, y + bh), color, 2)
        
        binary_mask_inv = cv2.bitwise_not(binary_mask)
        
        style_label, style_score, vlm_critique = self.vlm_analyzer.analyze(
            img_small, 
            subject_mask=binary_mask,
            custom_prompt_template=custom_analysis_prompt
        )

        face_points = []
        if self.pose_model:
            try:
                pose_results = self.pose_model(img_small, verbose=False)
                if pose_results and pose_results[0].keypoints is not None:
                    keypoints = pose_results[0].keypoints.data.cpu().numpy()
                    for person_kpts in keypoints:
                        if len(person_kpts) > 0 and person_kpts[0][2] > 0.5: face_points.append((int(person_kpts[0][0]), int(person_kpts[0][1])))
            except Exception: pass

        visual_elements, dist_contours = self._extract_visual_elements(img_small, binary_mask, valid_ocr_items, face_points, new_h, process_w)
        all_elements_mask = np.zeros((new_h, process_w), dtype=np.uint8)
        cv2.drawContours(all_elements_mask, dist_contours, -1, 255, -1)
        
        d_count_new, visual_order_score, vis_visual_order = self._analyze_distribution(img_small, visual_elements, dist_contours)
        d_count = d_count_new 
        
        # [NEW] Text Layout Analysis
        text_alignment_score, text_hierarchy_score, text_content_ratio, vis_text_design = self._analyze_text_layout(valid_ocr_items, process_w, new_h)
        
        # --- 构图分析 ---
        weighted_mass_map = binary_mask.astype(np.float32) / 255.0
        for item in valid_ocr_items:
            pts = item[0]
            cv2.fillPoly(weighted_mass_map, [pts], 2.0)
        for fp in face_points:
            cv2.circle(weighted_mass_map, fp, 20, 3.0, -1)
            
        saliency_vis = cv2.normalize(weighted_mass_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        comp_balance_score, comp_balance_center, vis_saliency_heatmap = self._calc_perceptual_balance(img_small, saliency_vis)
        
        # [Update] Capture dict from match_composition_template
        comp_layout_type, comp_layout_score, vis_layout_dict = self._match_composition_template(binary_mask, process_w, new_h, img_small)
        
        comp_negative_space_score, comp_negative_entropy = self._analyze_negative_space(binary_mask_inv)
        comp_visual_flow_score, comp_vanishing_point, vis_visual_flow = self._analyze_visual_flow(img_gray, binary_mask)

        # Symmetry
        try:
            k_blur = 31 if 31 % 2 != 0 else 32; img_blurred = cv2.GaussianBlur(img_small, (k_blur, k_blur), 0)
            cx_sym = process_w // 2; left_half = img_blurred[:, :cx_sym]; right_half = img_blurred[:, -cx_sym:]
            if left_half.shape == right_half.shape:
                diff_map = np.linalg.norm(left_half.astype(np.float32) - cv2.flip(right_half, 1).astype(np.float32), axis=2)
                score_symmetry = max(0, 100 * (1 - np.mean(diff_map) / 120.0))
                vis_symmetry_heatmap = cv2.cvtColor(cv2.applyColorMap(cv2.normalize(np.hstack((diff_map, cv2.flip(diff_map, 1))), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        except Exception: pass

        # Color stats
        dist_warm = np.cos(np.radians(h_ang - 40)); warm_signal = dist_warm * M
        vis_warmth = np.zeros_like(img_small); vis_warmth[warm_signal > 5.0] = [255, 0, 0]; vis_warmth[warm_signal < -5.0] = [0, 0, 255]
        warmth_ratio = float(np.count_nonzero(warm_signal > 5.0) / M.size)
        sat_mean = min(1.0, float(np.mean(M) / 60.0))
        vis_saturation = cv2.cvtColor(cv2.applyColorMap(cv2.normalize(M, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        bri_mean = float(np.mean(J) / 100.0); cont_std = float(np.std(J / 100.0))
        vis_brightness = cv2.cvtColor(cv2.normalize(J, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        vis_contrast = np.zeros_like(vis_brightness)
        clarity_mask = (J >= 70.0); clarity_ratio = float(np.count_nonzero(clarity_mask) / J.size)
        vis_clarity = img_small.copy(); vis_clarity[~clarity_mask] = (vis_clarity[~clarity_mask] * 0.3).astype(np.uint8)
        vis_clarity = cv2.cvtColor(vis_clarity, cv2.COLOR_BGR2RGB)

        total_px = binary_mask.size; fg_px = cv2.countNonZero(binary_mask); area_diff = abs((fg_px/total_px) - (1 - fg_px/total_px)) if total_px > 0 else 0
        if fg_px > 0 and (total_px - fg_px) > 0:
            rad_h = np.radians(h_ang); a_cam = C * np.cos(rad_h); b_cam = C * np.sin(rad_h)
            m_fg_J = np.mean(J[binary_mask > 0]); m_fg_a = np.mean(a_cam[binary_mask > 0]); m_fg_b = np.mean(b_cam[binary_mask > 0])
            m_bg_J = np.mean(J[binary_mask_inv > 0]); m_bg_a = np.mean(a_cam[binary_mask_inv > 0]); m_bg_b = np.mean(b_cam[binary_mask_inv > 0])
            color_diff = float(np.sqrt((m_fg_J - m_bg_J)**2 + (m_fg_a - m_bg_a)**2 + (m_fg_b - m_bg_b)**2))
            
            # [Optimization] 纹理对比算法
            kernel = np.ones((3, 3), np.uint8)
            mask_dilated = cv2.dilate(binary_mask, kernel, iterations=1)
            mask_eroded = cv2.erode(binary_mask, kernel, iterations=1)
            mask_boundary = cv2.bitwise_xor(mask_dilated, mask_eroded)
            
            valid_fg = cv2.bitwise_and(binary_mask, cv2.bitwise_not(mask_boundary))
            valid_bg = cv2.bitwise_and(binary_mask_inv, cv2.bitwise_not(mask_boundary))
            
            if cv2.countNonZero(valid_fg) < 50: valid_fg = binary_mask
            if cv2.countNonZero(valid_bg) < 50: valid_bg = binary_mask_inv

            J_32 = J.astype(np.float32)
            grad_x = cv2.Sobel(J_32, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(J_32, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(grad_x, grad_y)

            tex_fg = np.mean(magnitude[valid_fg > 0]) if cv2.countNonZero(valid_fg) > 0 else 0
            tex_bg = np.mean(magnitude[valid_bg > 0]) if cv2.countNonZero(valid_bg) > 0 else 0
            texture_diff = min(1.0, abs(tex_fg - tex_bg) / 50.0)
            
            # [Fix] Normalized color_diff (0-100)
            raw_color_diff = float(np.sqrt((m_fg_J - m_bg_J)**2 + (m_fg_a - m_bg_a)**2 + (m_fg_b - m_bg_b)**2))
            color_diff = min(100.0, (raw_color_diff / 150.0) * 100.0)

            mag_clip = np.clip(magnitude, 0, np.percentile(magnitude, 95))
            mag_vis = cv2.normalize(mag_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis_edge_composite = np.zeros((new_h, process_w, 3), dtype=np.uint8)
            
            vis_edge_composite[:, :, 0] = cv2.bitwise_and(mag_vis, mag_vis, mask=valid_fg)
            vis_edge_composite[:, :, 1] = cv2.bitwise_and(mag_vis, mag_vis, mask=valid_bg)
            
            dim_bg = (img_small * 0.3).astype(np.uint8)
            dim_bg_rgb = cv2.cvtColor(dim_bg, cv2.COLOR_BGR2RGB)
            mask_any_texture = cv2.bitwise_or(valid_fg, valid_bg)
            vis_edge_composite = np.where(mask_any_texture[:, :, None] > 0, vis_edge_composite, dim_bg_rgb)

            vis_color_contrast = np.zeros((300, 300, 3), dtype=np.uint8)
            vis_color_contrast[:] = list(cv2.mean(img_small, mask=binary_mask_inv)[:3])
            cv2.circle(vis_color_contrast, (150, 150), 100, list(cv2.mean(img_small, mask=binary_mask)[:3]), -1)
            vis_color_contrast = cv2.cvtColor(vis_color_contrast, cv2.COLOR_BGR2RGB)
        else:
             color_diff = 0.0; texture_diff = 0.0
             vis_edge_composite = None; vis_color_contrast = None

        # Text Legibility Analysis
        draw = ImageDraw.Draw(vis_pil); font = self._load_safe_font(16)
        text_scores = []; text_contrasts = []; detected_texts = []
        for (bbox, text_content, prob) in valid_ocr_items:
            detected_texts.append(text_content)
            pts = np.array(bbox, dtype=np.int32); bx, by, bw, bh = cv2.boundingRect(pts)
            if w_box := min(bw, process_w - bx) < 5 or (h_box := min(bh, new_h - by)) < 5: continue
            roi_g = img_gray[by:by+h_box, bx:bx+w_box]; roi_c = img_small[by:by+h_box, bx:bx+w_box]
            try:
                _, t_mask = cv2.threshold(roi_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if cv2.countNonZero(t_mask) > (w_box * h_box * 0.6): t_mask = cv2.bitwise_not(t_mask)
                m_txt = cv2.mean(roi_c, mask=t_mask)[:3]; m_bg = cv2.mean(roi_c, mask=cv2.bitwise_not(t_mask))[:3]
                
                # Raw contrast (BGR Euclidean distance, max ~441.67)
                raw_contrast = float(np.linalg.norm(np.array(m_txt) - np.array(m_bg)))
                
                # Calculate normalized legibility score (0-100)
                # s_con used for legibility score
                s_con = min(100, (raw_contrast / 441.67) * 100) 
                
                item_score = 0.7 * s_con + 0.3 * 80 
                text_scores.append(item_score); 
                
                # Store normalized contrast
                text_contrasts.append(s_con)
                
                color = (0, 255, 0) if item_score > 60 else (255, 0, 0)
                draw.rectangle([bx, by, bx+w_box, by+h_box], outline=color, width=2)
                draw.text((bx, by), f"S:{int(item_score)}", fill=(255, 255, 255), font=font)
            except Exception: continue
        
        vis_text_final = np.array(vis_pil)
        has_text = len(text_scores) > 0
        avg_text_score = float(np.mean(text_scores)) if has_text else 0.0
        avg_text_contrast = float(np.mean(text_contrasts)) if has_text else 0.0
        text_content_str = " | ".join(detected_texts) if has_text else "None"

        return OmniReport(
            comp_balance_score=round(comp_balance_score, 1),
            comp_balance_center=comp_balance_center,
            comp_layout_type=comp_layout_type,
            comp_layout_score=round(comp_layout_score, 1),
            comp_negative_space_score=round(comp_negative_space_score, 1),
            comp_negative_entropy=round(comp_negative_entropy, 3),
            comp_visual_flow_score=round(comp_visual_flow_score, 1),
            comp_visual_order_score=round(visual_order_score, 1),
            comp_vanishing_point=comp_vanishing_point,
            
            # Text Design
            text_alignment_score=round(text_alignment_score, 1),
            text_hierarchy_score=round(text_hierarchy_score, 1),
            text_content_ratio=round(text_content_ratio, 1),
            
            # Legacy fields - init with 0 or dummy
            composition_diagonal=0.0, 
            composition_thirds=0.0,
            composition_balance=0.0,
            composition_symmetry=0.0,
            
            color_warmth=round(warmth_ratio, 2), color_saturation=round(sat_mean, 2),
            color_brightness=round(bri_mean, 2), color_contrast=round(cont_std, 2),
            color_clarity=round(clarity_ratio, 2), color_harmony=round(harmony_score, 1),
            kobayashi_tags=kobayashi_tags, 
            semantic_style=style_label, semantic_score=round(style_score, 1), 
            vlm_critique=vlm_critique, 
            fg_area_diff=round(area_diff, 2), fg_color_diff=round(color_diff, 1), fg_texture_diff=round(texture_diff, 3),
            fg_text_present=has_text, fg_text_legibility=round(avg_text_score, 1),
            fg_text_contrast=round(avg_text_contrast, 1), fg_text_content=text_content_str,
            
            # Use initialized variables
            dist_count=int(d_count), dist_entropy=float(d_ent), dist_cv=float(d_cv),
            dist_size_cv=float(d_size_cv), dist_angle_entropy=float(d_angle_ent),
            
            vis_mask=vis_mask_debug, vis_all_elements=all_elements_mask,
            # Use initialized or calculated visualization variables
            vis_dist_entropy=vis_dist_entropy, vis_dist_size=vis_dist_size, vis_dist_angle=vis_dist_angle,
            
            vis_edge_fg=vis_edge_composite, vis_edge_bg=vis_edge_composite, vis_edge_composite=vis_edge_composite,
            vis_text_analysis=vis_text_final, vis_color_contrast=vis_color_contrast,
            vis_symmetry_heatmap=vis_symmetry_heatmap, vis_diag=vis_diag,
            vis_thirds=vis_thirds, vis_balance=vis_balance,
            vis_clarity=vis_clarity, vis_warmth=vis_warmth, vis_saturation=vis_saturation,
            vis_brightness=vis_brightness, vis_contrast=vis_contrast, vis_color_harmony=vis_harmony,
            
            vis_saliency_heatmap=vis_saliency_heatmap,
            vis_layout_template=None,
            vis_layout_dict=vis_layout_dict,
            vis_visual_flow=vis_visual_flow,
            vis_visual_order=vis_visual_order,
            vis_text_design=vis_text_design
        )

class AestheticDiagnostician:
    @staticmethod
    def generate_report(data: OmniReport, config: Dict = None) -> dict:
        return {"total_score": data.semantic_score if data.semantic_score > 0 else 75.0, "rating_level": "Good"}

class BenchmarkManager:
    def __init__(self):
        # 定义每个指标的评分策略
        self.metric_policies = {
            'color_clarity': 'sigmoid',
            'color_harmony': 'sigmoid',
            'comp_layout_score': 'sigmoid',
            'comp_visual_flow_score': 'sigmoid',
            'comp_visual_order_score': 'sigmoid',
            'text_alignment_score': 'sigmoid',
            'text_hierarchy_score': 'sigmoid',
            'fg_text_legibility': 'sigmoid',
            'fg_text_contrast': 'sigmoid', # [Add] New metric policy
            
            'color_warmth': 'gaussian',
            'color_saturation': 'gaussian',
            'color_brightness': 'gaussian',
            'color_contrast': 'gaussian',
            'comp_balance_score': 'gaussian',
            'comp_negative_space_score': 'gaussian',
            'fg_area_diff': 'gaussian',
            'fg_color_diff': 'gaussian',
            'fg_texture_diff': 'gaussian',
            'text_content_ratio': 'gaussian',
            
            'comp_negative_entropy': 'penalty' 
        }

    def _score_sigmoid(self, x, target, tolerance):
        diff = x - target
        if diff >= 0:
            return 100.0
        tol = max(1e-5, tolerance)
        k = 4.0 / tol 
        score = 100.0 / (1.0 + math.exp(-k * (diff + tol))) 
        return min(100.0, max(0.0, score))

    def _score_gaussian(self, x, target, tolerance):
        if tolerance < 1e-5: tolerance = 1e-5
        delta = x - target
        score = 100.0 * math.exp(-0.5 * (delta / tolerance) ** 2)
        return min(100.0, max(0.0, score))

    def _score_penalty(self, x, threshold, tolerance):
        if x <= threshold:
            return 100.0
        diff = x - threshold
        tol = max(1e-5, tolerance)
        score = 100.0 * math.exp(-1.0 * (diff / tol))
        return min(100.0, max(0.0, score))

    def create_profile(self, reports: List[OmniReport]) -> Dict:
        if not reports: return {}
        
        profile = {}
        all_keys = self.metric_policies.keys()
        
        data_matrix = {k: [] for k in all_keys}
        
        for r in reports:
            for k in all_keys:
                val = getattr(r, k, 0)
                if val is None: val = 0.0
                
                # [Fix] Only scale 0-1 metrics. fg_color_diff and fg_text_contrast are now pre-normalized.
                if k in ['color_warmth', 'color_saturation', 'color_brightness', 'color_contrast', 'color_clarity', 
                         'fg_area_diff', 'fg_texture_diff', 'text_content_ratio']:
                     val = val * 100.0
                elif k == 'comp_negative_entropy':
                     val = val * 100.0
                
                if k in ['fg_text_legibility', 'fg_text_contrast'] and not getattr(r, 'fg_text_present', False):
                    continue
                    
                data_matrix[k].append(val)
        
        for k, values in data_matrix.items():
            if not values:
                profile[k] = {'target': 0, 'tolerance': 10}
                continue
                
            mean = float(np.mean(values))
            std = float(np.std(values))
            
            tolerance = max(std * 1.5, 5.0) 
            
            if self.metric_policies[k] == 'penalty':
                mean = mean + std 
            
            profile[k] = {
                'target': mean,
                'tolerance': tolerance
            }
            
        return profile

    def score_against_benchmark(self, data: OmniReport, profile: Dict) -> Dict:
        if not profile:
            return {"total_score": 0, "rating_level": "N/A", "details": {}}
        
        total_score = 0.0
        total_weight = 0.0
        details = {}
        
        weights_config = profile.get('weights', {})
        
        for k, policy in self.metric_policies.items():
            val = getattr(data, k, 0)
            if val is None: val = 0.0
            
            if k in ['color_warmth', 'color_saturation', 'color_brightness', 'color_contrast', 'color_clarity', 
                     'fg_area_diff', 'fg_texture_diff', 'text_content_ratio', 'comp_negative_entropy']:
                 val = val * 100.0
            
            if k not in profile: continue 
            
            if k in ['fg_text_legibility', 'fg_text_contrast'] and not getattr(data, 'fg_text_present', False):
                details[k] = {'score': 0, 'actual': 0, 'target': 0, 'policy': policy}
                continue

            target = profile[k]['target']
            tolerance = profile[k].get('tolerance', 10.0)
            weight = weights_config.get(k, 1.0)
            
            if policy == 'sigmoid':
                item_score = self._score_sigmoid(val, target, tolerance)
            elif policy == 'gaussian':
                item_score = self._score_gaussian(val, target, tolerance)
            elif policy == 'penalty':
                item_score = self._score_penalty(val, target, tolerance)
            else:
                item_score = 0.0
                
            total_score += item_score * weight
            total_weight += weight
            
            details[k] = {
                'score': round(item_score, 1),
                'actual': round(val, 1),
                'target': round(target, 1),
                'tolerance': round(tolerance, 1),
                'weight': round(weight, 1),
                'policy': policy
            }
            
        final_score = total_score / total_weight if total_weight > 0 else 0
        
        if final_score >= 90: rating = "S (卓越)"
        elif final_score >= 80: rating = "A (优秀)"
        elif final_score >= 70: rating = "B (良好)"
        elif final_score >= 60: rating = "C (合格)"
        else: rating = "D (不合格)"
        
        return {
            "total_score": round(final_score, 1), 
            "rating_level": rating, 
            "details": details
        }