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

    def _calculate_auto_weights(self, reports_pos, reports_neg=None):
        """
        内部工具：自动权重推演 (FDR 策略)
        """
        all_dims = [
            'comp_balance_score', 'comp_layout_score', 'comp_negative_space_score', 
            'comp_visual_flow_score', 'comp_visual_order_score',
            'color_saturation', 'color_brightness', 'color_warmth', 'color_contrast', 'color_clarity', 'color_harmony',
            'fg_color_diff', 'fg_area_diff', 'fg_texture_diff', 'fg_text_legibility', 'fg_text_contrast',
            'text_alignment_score', 'text_hierarchy_score', 'text_content_ratio'
        ]
        
        suggested_weights = {}
        
        def get_values(reports, key):
            vals = []
            for r in reports:
                val = getattr(r, key, 0)
                if key in ['fg_text_legibility', 'fg_text_contrast'] and not getattr(r, 'fg_text_present', False): 
                    continue
                if val is not None:
                    vals.append(float(val))
            return vals

        use_fdr = reports_neg is not None and len(reports_neg) >= 2 and len(reports_pos) >= 2
        
        for k in all_dims:
            vals_pos = get_values(reports_pos, k)
            
            if not vals_pos or len(vals_pos) < 2:
                suggested_weights[k] = 1.0
                continue
            
            weight = 1.0
            
            if use_fdr:
                vals_neg = get_values(reports_neg, k)
                if not vals_neg or len(vals_neg) < 2:
                    use_fallback = True
                else:
                    use_fallback = False
                    mu_pos = np.mean(vals_pos)
                    mu_neg = np.mean(vals_neg)
                    var_pos = np.var(vals_pos)
                    var_neg = np.var(vals_neg)
                    epsilon = 1e-6
                    denominator = var_pos + var_neg + epsilon
                    fdr = ((mu_pos - mu_neg) ** 2) / denominator
                    weight = 0.5 + (fdr * 1.0)
                    weight = max(0.5, min(6.0, weight))
            
            if not use_fdr or use_fallback:
                std_dev = np.std(vals_pos)
                is_100_scale = np.max(vals_pos) > 1.5
                if is_100_scale:
                    std_norm = std_dev / 100.0
                else:
                    std_norm = std_dev
                weight = 0.2 / (std_norm + 0.05)
                weight = max(0.5, min(4.0, weight))
            
            suggested_weights[k] = round(weight, 1)
            
        return suggested_weights

    def _extract_distribution_data(self, reports):
        """内部工具：提取用于画图的原始分布数据 (18 指标)"""
        dist_data = {}
        all_dims = [
            'comp_balance_score', 'comp_layout_score', 'comp_negative_space_score', 
            'comp_visual_flow_score', 'comp_visual_order_score',
            'color_saturation', 'color_brightness', 'color_warmth', 'color_contrast', 'color_clarity', 'color_harmony',
            'fg_color_diff', 'fg_area_diff', 'fg_texture_diff', 'fg_text_legibility', 'fg_text_contrast',
            'text_alignment_score', 'text_hierarchy_score', 'text_content_ratio'
        ]
        
        for k in all_dims:
            vals = []
            for r in reports:
                v = getattr(r, k, 0)
                if v is None: v = 0
                
                # 1. 0-100 的指标 (无需处理)
                # [Fix] 这里的 text_content_ratio 已经是 0-100 了，移入此列
                if k in ['comp_balance_score', 'comp_layout_score', 'comp_negative_space_score', 
                         'comp_visual_flow_score', 'comp_visual_order_score', 
                         'color_harmony', 'fg_text_legibility', 'fg_text_contrast', 'fg_color_diff',
                         'text_alignment_score', 'text_hierarchy_score', 'text_content_ratio']:
                     vals.append(v)
                
                # 2. 0-1 的指标 (需要 * 100)
                elif k in ['color_warmth', 'color_saturation', 'color_brightness', 'color_clarity', 
                           'fg_area_diff', 'fg_texture_diff']:
                    vals.append(v * 100.0)
                
                # 3. 特殊指标 (对比度)
                elif k == 'color_contrast':
                    vals.append(min(100.0, (v / 0.3) * 100.0))
                
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
        
        if auto_weight_enable:
            auto_weights = self._calculate_auto_weights(reps_pos, reps_neg if reps_neg else None)
            final_weights.update(auto_weights)

        dist_data_pos = self._extract_distribution_data(reps_pos)
        dist_data_neg = self._extract_distribution_data(reps_neg) if reps_neg else {}

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
        
        return final_profile, {"pos": dist_data_pos, "neg": dist_data_neg}, stats