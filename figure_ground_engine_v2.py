import cv2
import numpy as np
from rembg import remove
from mask_utils import get_subject_mask_rembg
import easyocr
from dataclasses import dataclass, field
from typing import List, Optional

# === 数据结构定义 ===

@dataclass
class GeneralFigureGroundMetrics:
    """通用图形主体的图底指标"""
    area_diff: float        # 面积差异
    color_diff: float       # 色彩差异
    texture_diff: float     # 纹理差异
    is_strong: bool         # 综合判定是否强图底关系

@dataclass
class TextRegionMetrics:
    """单个文字块的图底指标"""
    text: str
    box: List[tuple]        # [(x1,y1), (x2,y2)]
    local_contrast: float   # 局部色彩对比度 (欧氏距离)
    bg_noise: float         # 背景纹理干扰度 (0-1)
    is_legible: bool        # 是否易读

@dataclass
class ComprehensiveReport:
    """综合分析报告"""
    general: GeneralFigureGroundMetrics
    text_regions: List[TextRegionMetrics]
    overall_score: float    # 全局打分
    visualization: np.ndarray

# === 分析引擎类 ===

class FigureGroundEngineV2:
    def __init__(self):
        # 1. 初始化通用分割模型 (rembg 自动加载)
        # 2. 初始化 OCR 模型 (首次运行会下载)
        print("Loading OCR Model...")
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        
    def analyze(self, image_input: np.ndarray) -> ComprehensiveReport:
        h, w = image_input.shape[:2]
        vis_img = image_input.copy()

        # --- A. 通用图底分析 (图形主体) ---
        gen_metrics = self._analyze_general_fg(image_input)
        
        # --- B. 文字图底分析 (信息传达) ---
        text_metrics = self._analyze_text_fg(image_input, vis_img)
        
        # --- C. 综合评分 ---
        # 简单加权：图形关系占 60%，文字关系占 40% (如果有文字)
        score = 0
        if text_metrics:
            # 计算文字的平均易读性分数 (对比度归一化到 0-100)
            avg_text_score = np.mean([min(100, t.local_contrast) for t in text_metrics])
            # 图形分数 (色彩差异 > 100 为满分)
            gen_score = min(100, gen_metrics.color_diff)
            score = 0.6 * gen_score + 0.4 * avg_text_score
        else:
            score = min(100, gen_metrics.color_diff)

        return ComprehensiveReport(
            general=gen_metrics,
            text_regions=text_metrics,
            overall_score=round(score, 1),
            visualization=vis_img
        )

    def _analyze_general_fg(self, img: np.ndarray) -> GeneralFigureGroundMetrics:
        """分析大图形主体的图底关系 (复用之前的逻辑)"""
        # 1. Rembg 分割
        binary_mask = get_subject_mask_rembg(img)
        binary_mask_inv = cv2.bitwise_not(binary_mask)
        
        # 2. 计算指标 (简化版)
        mean_fg = cv2.mean(img, mask=binary_mask)[:3]
        mean_bg = cv2.mean(img, mask=binary_mask_inv)[:3]
        color_diff = np.linalg.norm(np.array(mean_fg) - np.array(mean_bg))
        
        total_pixels = img.shape[0] * img.shape[1]
        fg_pixels = cv2.countNonZero(binary_mask)
        area_diff = abs((fg_pixels / total_pixels) - (1 - fg_pixels / total_pixels))
        
        # 3. 纹理差异 (Canny 密度差)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        fg_edge_d = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=binary_mask)) / (fg_pixels + 1)
        bg_edge_d = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=binary_mask_inv)) / (total_pixels - fg_pixels + 1)
        texture_diff = abs(fg_edge_d - bg_edge_d)

        return GeneralFigureGroundMetrics(
            area_diff=round(area_diff, 2),
            color_diff=round(color_diff, 1),
            texture_diff=round(texture_diff, 3),
            is_strong=(color_diff > 60 and area_diff > 0.3)
        )

    def _analyze_text_fg(self, img: np.ndarray, vis_img: np.ndarray) -> List[TextRegionMetrics]:
        """分析文字区域的局部图底关系"""
        results = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # OCR 检测
        ocr_res = self.reader.readtext(img)
        
        for (bbox, text, prob) in ocr_res:
            # 坐标处理
            pts = np.array(bbox, dtype=np.int32)
            x, y, w_box, h_box = cv2.boundingRect(pts)
            
            # 边界保护
            h_img, w_img = img.shape[:2]
            x = max(0, x); y = max(0, y)
            w_box = min(w_box, w_img - x); h_box = min(h_box, h_img - y)
            if w_box < 5 or h_box < 5: continue

            # --- 局部精细分割 (Otsu) ---
            roi_gray = gray[y:y+h_box, x:x+w_box]
            roi_color = img[y:y+h_box, x:x+w_box]
            
            # 自动阈值提取笔画
            _, text_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 判断背景色 (采样四角) 以确定 mask 是否需要反转
            corners = np.concatenate([
                text_mask[0:2, :], text_mask[-2:, :], 
                text_mask[:, 0:2], text_mask[:, -2:]
            ], axis=None)
            if np.mean(corners) > 127: # 背景是白的
                text_mask = cv2.bitwise_not(text_mask) # 让笔画变白(255)
            
            bg_mask = cv2.bitwise_not(text_mask)

            # --- 计算局部对比度 ---
            mean_text = cv2.mean(roi_color, mask=text_mask)[:3]
            mean_bg = cv2.mean(roi_color, mask=bg_mask)[:3]
            local_contrast = np.linalg.norm(np.array(mean_text) - np.array(mean_bg))

            # --- 计算背景杂乱度 ---
            # 计算 ROI 内背景区域的边缘密度
            roi_edges = cv2.Canny(roi_gray, 50, 150)
            bg_edge_pixels = cv2.countNonZero(cv2.bitwise_and(roi_edges, roi_edges, mask=bg_mask))
            bg_area = cv2.countNonZero(bg_mask)
            bg_noise = bg_edge_pixels / bg_area if bg_area > 0 else 0

            # --- 判定与绘图 ---
            # 易读标准：对比度 > 70 且 背景杂乱度 < 0.2
            is_legible = local_contrast > 70 and bg_noise < 0.2
            
            results.append(TextRegionMetrics(
                text=text, box=[(x, y), (x+w_box, y+h_box)],
                local_contrast=round(local_contrast, 1),
                bg_noise=round(bg_noise, 3),
                is_legible=is_legible
            ))

            # 可视化：绿框表示好，红框表示差
            color = (0, 255, 0) if is_legible else (0, 0, 255)
            cv2.rectangle(vis_img, (x, y), (x+w_box, y+h_box), color, 2)
            # 标注对比度数值
            cv2.putText(vis_img, f"C:{int(local_contrast)}", (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return results