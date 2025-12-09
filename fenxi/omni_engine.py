import cv2
import numpy as np
import colour
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Union, Any
import platform
import os
from scipy.spatial import cKDTree
from scipy.stats import entropy
import math
import base64
import json
import re

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

# ==========================================
# 0. 提示词模板 (适配 Grounding 格式)
# ==========================================
DEFAULT_ANALYSIS_PROMPT = """请检测图像中所有属于 "main_subject（视觉主体）、text（文字区域）" 类别的物体。
对于每个物体，请提供其类别、边界框，如果是文字请提供内容（content）。
格式为 JSON 列表：
[
  {{"category": "main_subject", "bbox": "<bbox>x1 y1 x2 y2</bbox>"}},
  {{"category": "text", "bbox": "<bbox>x1 y1 x2 y2</bbox>", "content": "文字内容"}}
]
坐标 x1 y1 x2 y2 为 0-1000 的归一化整数，顺序为 左 上 右 下。"""

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
    # --- 构图与视觉秩序 (5) ---
    comp_balance_score: float
    comp_balance_center: Tuple[int, int]
    comp_layout_type: str
    comp_layout_score: float
    comp_negative_space_score: float
    comp_negative_entropy: float
    comp_visual_flow_score: float
    comp_visual_order_score: float
    comp_vanishing_point: Optional[Tuple[int, int]]
    
    # --- 色彩氛围 (6) ---
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

    # --- 图底与信息 (3) ---
    fg_area_diff: float
    fg_color_diff: float
    fg_texture_diff: float
    
    # --- 文字排版 (4) ---
    fg_text_present: bool
    fg_text_legibility: float
    fg_text_content: str
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
    vis_layout_template: Optional[np.ndarray] = None 
    vis_layout_dict: Optional[Dict] = None 
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
    
    composition_diagonal: float = 0.0
    composition_thirds: float = 0.0
    composition_balance: float = 0.0
    composition_symmetry: float = 0.0
    
    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.comp_balance_score, self.comp_layout_score, self.comp_negative_space_score, 
            self.comp_visual_flow_score, self.comp_visual_order_score,
            self.color_saturation * 100, self.color_brightness * 100, self.color_warmth * 100,
            (self.color_contrast / 0.3) * 100, self.color_clarity * 100, self.color_harmony,
            self.fg_color_diff, self.fg_area_diff * 100, self.fg_texture_diff * 100,
            self.fg_text_legibility if self.fg_text_present else 0.0, 
            self.text_alignment_score, self.text_hierarchy_score, self.text_content_ratio
        ], dtype=np.float32)

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

    def analyze(self, image_bgr, custom_prompt_template=None):
        if not self.client:
            return None

        base64_image = self._encode_image(image_bgr)
        h, w = image_bgr.shape[:2]
        
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
                        "content": [{"type": "text", "text": "开始检测。"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}],
                    },
                ],
                extra_body={
                    "reasoning_config": {
                        "mode": "disabled"
                    }
                }
            )
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            
            result_list = []
            try:
                match = re.search(r'(\[.*\]|\{.*\})', content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        result_list = parsed
                    elif isinstance(parsed, dict):
                        if "objects" in parsed and isinstance(parsed["objects"], list):
                            result_list = parsed["objects"]
                        elif "text_regions" in parsed and isinstance(parsed["text_regions"], list):
                             result_list = parsed.get("text_regions", [])
                             if "subject_rect_1000" in parsed:
                                 result_list.append({"category": "main_subject", "bbox": parsed["subject_rect_1000"]})
                        else:
                            result_list = [parsed]
                else:
                    print("DEBUG: No JSON structure found in VLM response.")
                    return None
            except json.JSONDecodeError as e:
                print(f"DEBUG: VLM JSON Decode Error: {e}")
                return None

            def parse_bbox_token(bbox_input: Union[str, List, Tuple]):
                if bbox_input is None: return None
                
                nums = []
                if isinstance(bbox_input, (list, tuple)):
                    flat = []
                    for x in bbox_input:
                        if isinstance(x, (int, float)): flat.append(x)
                        elif isinstance(x, str) and x.isdigit(): flat.append(int(x))
                    nums = flat
                elif isinstance(bbox_input, str):
                    nums = list(map(int, re.findall(r'\d+', bbox_input)))
                
                if len(nums) >= 4:
                    x1, y1, x2, y2 = nums[:4]
                    if x1 > x2: x1, x2 = x2, x1
                    if y1 > y2: y1, y2 = y2, y1
                    return [
                        int(x1 / 1000 * w), int(y1 / 1000 * h),
                        int(x2 / 1000 * w), int(y2 / 1000 * h)
                    ]
                return None

            vlm_data = {
                "subject_box": None,
                "text_items": []
            }
            
            for item in result_list:
                if not isinstance(item, dict): continue
                
                category = str(item.get("category", "")).lower()
                bbox_raw = item.get("bbox") or item.get("rect_1000") or item.get("box")
                
                box_px = parse_bbox_token(bbox_raw)
                if not box_px: continue
                
                x1, y1, x2, y2 = box_px
                final_box = [x1, y1, x2, y2]

                if any(k in category for k in ["main_subject", "subject", "主体", "object"]):
                    if vlm_data["subject_box"] is None:
                        vlm_data["subject_box"] = final_box
                    else:
                        curr_area = (final_box[2]-final_box[0]) * (final_box[3]-final_box[1])
                        prev_area = (vlm_data["subject_box"][2]-vlm_data["subject_box"][0]) * (vlm_data["subject_box"][3]-vlm_data["subject_box"][1])
                        if curr_area > prev_area:
                            vlm_data["subject_box"] = final_box
                
                content = item.get("content", "")
                is_text_cat = any(k in category for k in ["text", "ocr", "文字", "content"])
                
                if is_text_cat or (content and len(content) > 0):
                    poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    if not content: content = "Text"
                    vlm_data["text_items"].append((poly, content, 1.0))

            return vlm_data

        except Exception as e:
            print(f"VLM Analysis Error: {e}")
            return None

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

    def extract_main_subject_mask(self, image_bgr: np.ndarray, config: Dict = None, text_boxes: List = None, vlm_box_prompt: List = None) -> Tuple[np.ndarray, List]:
        """
        Modified: 
        1. VLM box is the primary source.
        2. 'box_prompts' (used for SAM/calculation) includes padding.
        3. 'debug_boxes' (used for visualization) stores the RAW box to align with visual expectations.
        """
        h, w = image_bgr.shape[:2]
        box_prompts = []
        debug_boxes = []
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Helper to handle distinct calculation vs display boxes
        def add_box_with_distinct_display(raw_box, label_type, padding_ratio=0.0):
            x1, y1, x2, y2 = raw_box
            bw = x2 - x1; bh = y2 - y1
            
            # 1. Calc Padded Box (for SAM / Masking)
            pad_x = int(bw * padding_ratio); pad_y = int(bh * padding_ratio)
            px1 = max(0, x1 - pad_x); py1 = max(0, y1 - pad_y)
            px2 = min(w, x2 + pad_x); py2 = min(h, y2 + pad_y)
            
            if px2 > px1 and py2 > py1:
                # Store Padded for Calculation
                box_prompts.append([px1, py1, px2, py2])
                # Store Raw for Display (Visualization)
                debug_boxes.append({'box': [x1, y1, bw, bh], 'type': label_type})

        # 1. [PRIMARY] VLM Guidance (Subject)
        #    - Calculation: Pad 2.5% (0.025) as requested
        #    - Display: Show Raw box to match VLM output
        if vlm_box_prompt is not None:
            vx1, vy1, vx2, vy2 = vlm_box_prompt
            if vx2 > vx1 and vy2 > vy1:
                add_box_with_distinct_display(vlm_box_prompt, 'vlm_subject', padding_ratio=0.025)
                
                # Fallback: If SAM missing, draw the PADDED box as mask (best guess for segmentation)
                if not self.sam_predictor:
                    # Retrieve the last padded box we just added
                    px1, py1, px2, py2 = box_prompts[-1]
                    cv2.rectangle(final_mask, (px1, py1), (px2, py2), 255, -1)
        
        # 2. [FALLBACK] U2-Net
        elif self.u2net_session:
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
                        # U2Net is imprecise, use default 5% padding
                        add_box_with_distinct_display([x, y, x+bw, y+bh], 'object_u2net', padding_ratio=0.05)
                        cv2.drawContours(u2net_valid_mask, [cnt], -1, 255, -1)
                final_mask = cv2.bitwise_or(final_mask, u2net_valid_mask)
            except Exception: pass

        # 3. Text Boxes
        #    - Calculation: No padding (0.0) to strictly segment text area
        #    - Display: Raw box
        if text_boxes:
            for bbox in text_boxes:
                pts = np.array(bbox, dtype=np.int32)
                x, y, bw, bh = cv2.boundingRect(pts)
                if bw > 2 and bh > 2:
                    add_box_with_distinct_display([x, y, x+bw, y+bh], 'text', padding_ratio=0.0)

        # 4. SAM Refinement
        if self.sam_predictor and box_prompts:
            try:
                self.sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                for i, box in enumerate(box_prompts):
                    input_box = np.array(box)
                    masks, _, _ = self.sam_predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False)
                    mask_uint8 = (masks[0] * 255).astype(np.uint8)
                    
                    # [Logic] Clip mask to the PADDED box to prevent bleeding into far background
                    # Since 'box' here IS the padded box, this ensures mask stays within the search area
                    px1, py1, px2, py2 = box
                    box_mask = np.zeros_like(mask_uint8)
                    cv2.rectangle(box_mask, (px1, py1), (px2, py2), 255, -1)
                    mask_uint8 = cv2.bitwise_and(mask_uint8, box_mask)
                    
                    # Guard: If SAM returned empty, use the box itself
                    if cv2.countNonZero(mask_uint8) == 0:
                         mask_uint8 = box_mask

                    final_mask = cv2.bitwise_or(final_mask, mask_uint8)
            except Exception as e: 
                print(f"SAM Error: {e}")
                pass
        
        return final_mask, debug_boxes

    def _fallback_saliency(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return mask

# === 全能视觉分析引擎 ===
class OmniVisualEngine:
    def __init__(self, vlm_api_key=None, vlm_endpoint=None):
        print("Initializing Omni Engine v25.0 (VLM Recognition Mode)...")
        self.segmenter = HybridSegmenter()
        self.pose_model = ModelRegistry.get_yolo_pose()
        self.cam02 = CIECAM02_Vectorized()
        self.vlm_analyzer = DoubaoVLMAnalyzer(api_key=vlm_api_key, model_endpoint=vlm_endpoint)

    def _load_safe_font(self, font_size=16):
        return ImageFont.load_default()

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

    def _calc_oklab_score(self, bgr_txt, bgr_bg):
        try:
            t_bgr = np.array(bgr_txt[:3], dtype=np.float32)
            b_bgr = np.array(bgr_bg[:3], dtype=np.float32)
            rgb_txt = t_bgr[::-1] / 255.0
            rgb_bg = b_bgr[::-1] / 255.0
            lab_txt = colour.convert(rgb_txt, 'sRGB', 'Oklab')
            lab_bg = colour.convert(rgb_bg, 'sRGB', 'Oklab')
            distance = float(np.linalg.norm(lab_txt - lab_bg))
            score = min(100.0, distance * 100.0)
            return score
        except Exception as e:
            try:
                t_bgr = np.array(bgr_txt[:3], dtype=np.float32)
                b_bgr = np.array(bgr_bg[:3], dtype=np.float32)
                return float(np.mean(np.abs(t_bgr - b_bgr)) / 2.55)
            except:
                return 0.0

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
        
        boxes = [np.array(item[0], dtype=np.int32) for item in valid_ocr_items]
        
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
        
        # [Updated] Use new VLM logic
        # 1. Image Preprocessing
        h, w = image_input.shape[:2]; scale = process_w / w; new_h = int(h * scale)
        img_small = cv2.resize(image_input, (process_w, new_h))
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB); img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        vis_pil = Image.fromarray(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
        
        # 2. VLM Recognition (Primary Source of Truth)
        # [Fix] Get prompt from config
        custom_analysis_prompt = config.get('analysis_prompt')
        vlm_result = self.vlm_analyzer.analyze(img_small, custom_prompt_template=custom_analysis_prompt)
        
        valid_ocr_items = []
        vlm_subject_box = None
        
        if vlm_result:
            vlm_subject_box = vlm_result.get("subject_box")
            valid_ocr_items = vlm_result.get("text_items", [])
        else:
            print("⚠️ VLM Recognition Failed. Analysis may be incomplete.")
            
        text_boxes_low_conf = [item[0] for item in valid_ocr_items]
        
        # 3. Segmentation (Guided by VLM)
        binary_mask, debug_boxes = self.segmenter.extract_main_subject_mask(
            img_small, config, 
            text_boxes=text_boxes_low_conf,
            vlm_box_prompt=vlm_subject_box 
        )
        binary_mask_inv = cv2.bitwise_not(binary_mask)
        
        # 4. Standard Metric Analysis (Using VLM-derived data)
        
        # --- Color Stats (CIECAM02) ---
        img_xyz = cv2.cvtColor(img_small, cv2.COLOR_BGR2XYZ).astype(np.float32); img_xyz_norm = img_xyz / 255.0 * 100.0
        cam_res = self.cam02.forward(img_xyz_norm); J, C, h_ang, M = cam_res['J'], cam_res['C'], cam_res['h'], cam_res['M']
        harmony_score, vis_harmony = self._analyze_color_harmony(h_ang, C, J, M, img_small)
        kobayashi_tags = self._analyze_kobayashi_image_scale(h_ang, C, J, M)
        
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

        # --- Figure/Ground ---
        total_px = binary_mask.size; fg_px = cv2.countNonZero(binary_mask); area_diff = abs((fg_px/total_px) - (1 - fg_px/total_px)) if total_px > 0 else 0
        if fg_px > 0 and (total_px - fg_px) > 0:
            m_fg_bgr = cv2.mean(img_small, mask=binary_mask)[:3]
            m_bg_bgr = cv2.mean(img_small, mask=binary_mask_inv)[:3]
            color_diff = self._calc_oklab_score(m_fg_bgr, m_bg_bgr)
            
            # Texture
            J_32 = J.astype(np.float32)
            grad_x = cv2.Sobel(J_32, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(J_32, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(grad_x, grad_y)
            tex_fg = np.mean(magnitude[binary_mask > 0])
            tex_bg = np.mean(magnitude[binary_mask_inv > 0])
            texture_diff = min(1.0, abs(tex_fg - tex_bg) / 50.0)
            
            mag_clip = np.clip(magnitude, 0, np.percentile(magnitude, 95))
            mag_vis = cv2.normalize(mag_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis_edge_composite = np.zeros((new_h, process_w, 3), dtype=np.uint8)
            vis_edge_composite[:, :, 0] = cv2.bitwise_and(mag_vis, mag_vis, mask=binary_mask)
            vis_edge_composite[:, :, 1] = cv2.bitwise_and(mag_vis, mag_vis, mask=binary_mask_inv)
            vis_color_contrast = np.zeros((300, 300, 3), dtype=np.uint8)
            vis_color_contrast[:] = list(m_bg_bgr)
            cv2.circle(vis_color_contrast, (150, 150), 100, list(m_fg_bgr), -1)
            vis_color_contrast = cv2.cvtColor(vis_color_contrast, cv2.COLOR_BGR2RGB)
        else:
             color_diff = 0.0; texture_diff = 0.0
             vis_edge_composite = None; vis_color_contrast = None

        # --- Text Analysis (Based on VLM output) ---
        draw = ImageDraw.Draw(vis_pil); font = self._load_safe_font(16)
        text_scores = []; detected_texts = []
        for (poly, text_content, _) in valid_ocr_items:
            detected_texts.append(text_content)
            pts = np.array(poly, dtype=np.int32)
            bx, by, bw, bh = cv2.boundingRect(pts)
            
            # [Fix] 修复赋值逻辑 Bug：避免 := 优先级问题导致 w_box 变成 boolean
            w_box = min(bw, process_w - bx)
            h_box = min(bh, new_h - by)
            if w_box < 3 or h_box < 3: continue
            
            # Local contrast calculation
            roi_c = img_small[by:by+h_box, bx:bx+w_box]
            roi_g = img_gray[by:by+h_box, bx:bx+w_box]
            try:
                _, t_mask = cv2.threshold(roi_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                m_txt = cv2.mean(roi_c, mask=t_mask)
                m_bg = cv2.mean(roi_c, mask=cv2.bitwise_not(t_mask))
                item_score = self._calc_oklab_score(m_txt, m_bg)
                text_scores.append(item_score)
                
                color = (0, 255, 0) if item_score > 60 else (255, 0, 0)
                draw.rectangle([bx, by, bx+w_box, by+h_box], outline=color, width=2)
                draw.text((bx, by), f"S:{int(item_score)}", fill=(255, 255, 255), font=font)
            except Exception: continue
            
        vis_text_final = np.array(vis_pil)
        has_text = len(text_scores) > 0
        avg_text_score = float(np.mean(text_scores)) if has_text else 0.0
        text_content_str = " | ".join(detected_texts) if has_text else "None"
        
        text_alignment_score, text_hierarchy_score, text_content_ratio, vis_text_design = self._analyze_text_layout(valid_ocr_items, process_w, new_h)

        # --- Composition ---
        face_points = [] # Deprecated but kept for compatibility
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
        
        dist_count, visual_order_score, vis_visual_order = self._analyze_distribution(img_small, visual_elements, dist_contours)
        
        weighted_mass_map = binary_mask.astype(np.float32) / 255.0
        saliency_vis = cv2.normalize(weighted_mass_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        comp_balance_score, comp_balance_center, vis_saliency_heatmap = self._calc_perceptual_balance(img_small, saliency_vis)
        comp_layout_type, comp_layout_score, vis_layout_dict = self._match_composition_template(binary_mask, process_w, new_h, img_small)
        comp_negative_space_score, comp_negative_entropy = self._analyze_negative_space(binary_mask_inv)
        comp_visual_flow_score, comp_vanishing_point, vis_visual_flow = self._analyze_visual_flow(img_gray, binary_mask)

        # Symmetry
        score_symmetry = 0.0; vis_symmetry_heatmap = None
        try:
            k_blur = 31 if 31 % 2 != 0 else 32; img_blurred = cv2.GaussianBlur(img_small, (k_blur, k_blur), 0)
            cx_sym = process_w // 2; left_half = img_blurred[:, :cx_sym]; right_half = img_blurred[:, -cx_sym:]
            if left_half.shape == right_half.shape:
                diff_map = np.linalg.norm(left_half.astype(np.float32) - cv2.flip(right_half, 1).astype(np.float32), axis=2)
                score_symmetry = max(0, 100 * (1 - np.mean(diff_map) / 120.0))
                vis_symmetry_heatmap = cv2.cvtColor(cv2.applyColorMap(cv2.normalize(np.hstack((diff_map, cv2.flip(diff_map, 1))), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        except Exception: pass

        # Visualization placeholders for legacy props
        vis_mask_debug = img_small.copy()
        mask_indices = binary_mask > 0
        if np.any(mask_indices):
            overlay = vis_mask_debug.copy()
            overlay[mask_indices] = (0, 0, 255) 
            vis_mask_debug = cv2.addWeighted(vis_mask_debug, 0.7, overlay, 0.3, 0)
        for item in debug_boxes:
            box = item['box']
            x, y, bw, bh = box
            color = (0, 255, 0) if item['type'] == 'vlm_subject' else (255, 0, 0)
            cv2.rectangle(vis_mask_debug, (x, y), (x + bw, y + bh), color, 2)

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
            
            text_alignment_score=round(text_alignment_score, 1),
            text_hierarchy_score=round(text_hierarchy_score, 1),
            text_content_ratio=round(text_content_ratio, 1),
            
            composition_diagonal=0.0, composition_thirds=0.0, composition_balance=0.0, composition_symmetry=round(score_symmetry,1),
            
            color_warmth=round(warmth_ratio, 2), color_saturation=round(sat_mean, 2),
            color_brightness=round(bri_mean, 2), color_contrast=round(cont_std, 2),
            color_clarity=round(clarity_ratio, 2), color_harmony=round(harmony_score, 1),
            kobayashi_tags=kobayashi_tags, 
            
            # VLM Semantic fields set to Defaults
            semantic_style="N/A", 
            semantic_score=0.0, 
            vlm_critique="VLM 功能仅用于文字与主体识别", 
            
            fg_area_diff=round(area_diff, 2), fg_color_diff=round(color_diff, 1), fg_texture_diff=round(texture_diff, 3),
            
            fg_text_present=has_text, 
            fg_text_legibility=round(avg_text_score, 1),
            fg_text_content=text_content_str,
            
            dist_count=int(dist_count), dist_entropy=0.0, dist_cv=0.0,
            dist_size_cv=0.0, dist_angle_entropy=0.0,
            
            vis_mask=vis_mask_debug, vis_all_elements=all_elements_mask,
            vis_dist_entropy=None, vis_dist_size=None, vis_dist_angle=None,
            
            vis_edge_fg=vis_edge_composite, vis_edge_bg=vis_edge_composite, vis_edge_composite=vis_edge_composite,
            vis_text_analysis=vis_text_final, vis_color_contrast=vis_color_contrast,
            vis_symmetry_heatmap=vis_symmetry_heatmap, vis_diag=None,
            vis_thirds=None, vis_balance=None,
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
        return {"total_score": 85.0, "rating_level": "Good"} # Static default

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
            
            'fg_color_diff': 'gaussian',
            'fg_texture_diff': 'gaussian',

            'color_warmth': 'gaussian',
            'color_saturation': 'gaussian',
            'color_brightness': 'gaussian',
            'color_contrast': 'gaussian',
            'comp_balance_score': 'gaussian',
            'comp_negative_space_score': 'gaussian',
            'fg_area_diff': 'gaussian',
            
            'text_content_ratio': 'gaussian',
            
            'comp_negative_entropy': 'penalty' 
        }

    def _score_sigmoid(self, x, target, std):
        diff = x - target
        if diff >= 0:
            return 100.0
        sigma = max(1e-5, std)
        k = 4.0 / sigma
        score = 100.0 / (1.0 + math.exp(-k * (diff + sigma))) 
        return min(100.0, max(0.0, score))

    def _score_gaussian(self, x, target, std):
        sigma = max(1e-5, std)
        delta = x - target
        score = 100.0 * math.exp(-0.5 * (delta / sigma) ** 2)
        return min(100.0, max(0.0, score))

    def _score_penalty(self, x, threshold, std):
        if x <= threshold:
            return 100.0
        diff = x - threshold
        sigma = max(1e-5, std)
        score = 100.0 * math.exp(-1.0 * (diff / sigma))
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
                
                # [Fix] Only scale 0-1 metrics. fg_color_diff is now pre-normalized.
                if k in ['color_warmth', 'color_saturation', 'color_brightness', 'color_contrast', 'color_clarity', 
                         'fg_area_diff', 'fg_texture_diff', 'comp_negative_entropy']:
                     val = val * 100.0
                
                if k in ['fg_text_legibility'] and not getattr(r, 'fg_text_present', False):
                    continue
                    
                data_matrix[k].append(val)
        
        for k, values in data_matrix.items():
            if not values:
                profile[k] = {'target': 0, 'std': 10}
                continue
                
            mean = float(np.mean(values))
            std = float(np.std(values))
            
            std = max(std, 5.0)
            
            if self.metric_policies[k] == 'penalty':
                target_val = mean + std 
            else:
                target_val = mean
            
            profile[k] = {
                'target': target_val,
                'std': std
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
            
            # Apply same scaling as in create_profile
            if k in ['color_warmth', 'color_saturation', 'color_brightness', 'color_contrast', 'color_clarity', 
                     'fg_area_diff', 'fg_texture_diff', 'comp_negative_entropy']:
                 val = val * 100.0
            
            if k not in profile: continue 
            
            if k in ['fg_text_legibility'] and not getattr(data, 'fg_text_present', False):
                details[k] = {'score': 0, 'actual': 0, 'target': 0, 'policy': policy}
                continue

            target = profile[k]['target']
            std = profile[k].get('std', 10.0)
            weight = weights_config.get(k, 1.0)
            
            if policy == 'sigmoid':
                item_score = self._score_sigmoid(val, target, std)
            elif policy == 'gaussian':
                item_score = self._score_gaussian(val, target, std)
            elif policy == 'penalty':
                item_score = self._score_penalty(val, target, std)
            else:
                item_score = 0.0
                
            total_score += item_score * weight
            total_weight += weight
            
            details[k] = {
                'score': round(item_score, 1),
                'actual': round(val, 1),
                'target': round(target, 1),
                'std': round(std, 1),
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