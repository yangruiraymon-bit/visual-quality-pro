import cv2
import numpy as np
from omni_engine import OmniVisualEngine, BenchmarkManager

class BenchmarkTrainer:
    def __init__(self):
        self.engine = OmniVisualEngine()
        self.manager = BenchmarkManager()

    def _process_images(self, file_buffers, config):
        """内部工具：批量处理图片流"""
        reports = []
        for f in file_buffers:
            if hasattr(f, 'seek'): f.seek(0)
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                reports.append(self.engine.analyze(img, config=config))
        return reports

    def _calculate_auto_weights(self, reports):
        """内部工具：自动权重推演 (适配 15 核心指标)"""
        if not reports or len(reports) < 2: return {}
        
        all_dims = [
            'comp_balance_score', 'comp_layout_score', 'comp_negative_space_score', 
            'comp_visual_flow_score', 'comp_visual_order_score',
            
            'color_saturation', 'color_brightness', 'color_warmth', 'color_contrast', 'color_clarity', 'color_harmony',
            'fg_color_diff', 'fg_area_diff', 'fg_texture_diff', 'fg_text_legibility'
        ]
        
        suggested_weights = {}
        data_matrix = {k: [] for k in all_dims}
        for r in reports:
            for k in all_dims:
                val = getattr(r, k, 0) or 0
                if k == 'fg_text_legibility' and not getattr(r, 'fg_text_present', False): continue
                data_matrix[k].append(val)
        
        for k, values in data_matrix.items():
            if not values or len(values) < 2:
                suggested_weights[k] = 1.0; continue
            
            std_dev = np.std(values)
            
            # 已经归一化到 0-100 或 0-1 的指标处理不同
            if k in ['fg_color_diff', 'color_harmony', 'fg_text_legibility', 'comp_balance_score', 'comp_layout_score', 'comp_negative_space_score', 'comp_visual_flow_score', 'comp_visual_order_score']:
                std_norm = std_dev / 100.0
            else:
                std_norm = std_dev
            
            raw_w = 0.2 / (std_norm + 0.05)
            suggested_weights[k] = round(max(0.5, min(4.0, raw_w)), 1)
            
        return suggested_weights

    def _extract_distribution_data(self, reports):
        """内部工具：提取用于画图的原始分布数据 (15 指标)"""
        dist_data = {}
        all_dims = [
            'comp_balance_score', 'comp_layout_score', 'comp_negative_space_score', 
            'comp_visual_flow_score', 'comp_visual_order_score',

            'color_saturation', 'color_brightness', 'color_warmth', 'color_contrast', 'color_clarity', 'color_harmony',
            'fg_color_diff', 'fg_area_diff', 'fg_texture_diff', 'fg_text_legibility'
        ]
        for k in all_dims:
            vals = []
            for r in reports:
                v = getattr(r, k, 0)
                if v is None: v = 0
                
                # 已经是 0-100 的无需处理
                if k in ['comp_balance_score', 'comp_layout_score', 'comp_negative_space_score', 
                         'comp_visual_flow_score', 'comp_visual_order_score', 
                         'color_harmony', 'fg_text_legibility']:
                     vals.append(v)
                # 0-1 范围的需要乘 100
                elif k in ['color_warmth', 'color_saturation', 'color_brightness', 'color_clarity', 'fg_area_diff', 'fg_texture_diff']:
                    vals.append(v * 100)
                # 对比度特殊处理
                elif k == 'color_contrast':
                    vals.append(min(100, (v/0.3)*100))
                # 色差特殊处理
                elif k == 'fg_color_diff':
                    vals.append(min(100, v))
                else:
                    vals.append(v)
            dist_data[k] = vals
        return dist_data

    def train(self, pos_files, neg_files, config, auto_weight_enable=True):
        """
        对外接口：双向训练
        """
        reps_pos = self._process_images(pos_files, config)
        if not reps_pos: raise ValueError("没有有效的正向图片")
            
        prof_pos = self.manager.create_profile(reps_pos)
        
        prof_neg = {}
        reps_neg = []
        if neg_files:
            reps_neg = self._process_images(neg_files, config)
            if reps_neg: prof_neg = self.manager.create_profile(reps_neg)
        
        final_weights = config.get('weights', {}).copy()
        if auto_weight_enable and len(reps_pos) > 1:
            auto_weights = self._calculate_auto_weights(reps_pos)
            final_weights.update(auto_weights)

        dist_data = self._extract_distribution_data(reps_pos)

        final_tolerances = {}
        for k, v in prof_pos.items():
            if isinstance(v, dict) and 'tolerance' in v:
                final_tolerances[k] = v['tolerance']
        
        final_profile = {
            "positive": prof_pos,
            "negative": prof_neg,
            "weights": final_weights,
            "tolerances": final_tolerances
        }
        
        stats = {
            "pos_count": len(reps_pos),
            "neg_count": len(reps_neg),
            "auto_weighted": auto_weight_enable
        }
        
        return final_profile, dist_data, stats