import cv2
import numpy as np
from omni_engine import OmniVisualEngine, BenchmarkManager
from luv_analysis import LUVAnalysisEngine

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
        """内部工具：自动权重推演"""
        if not reports or len(reports) < 2: return {}
        
        all_dims = [
            'composition_diagonal', 'composition_thirds', 'composition_balance', 'composition_symmetry',
            'color_saturation', 'color_brightness', 'color_warmth', 'color_contrast', 'color_clarity',
            'fg_color_diff', 'fg_area_diff', 'fg_texture_diff', 'fg_text_legibility'
        ]
        
        suggested_weights = {}
        # 收集数据计算标准差
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
            is_large = k in ['fg_color_diff', 'composition_symmetry', 'fg_text_legibility', 'composition_diagonal', 'composition_thirds', 'composition_balance']
            std_norm = std_dev / 100.0 if is_large else std_dev
            raw_w = 0.2 / (std_norm + 0.05)
            suggested_weights[k] = round(max(0.5, min(4.0, raw_w)), 1)
            
        return suggested_weights

    def _extract_distribution_data(self, reports):
        """【新增】内部工具：提取用于画图的原始分布数据"""
        dist_data = {}
        all_dims = [
            'composition_diagonal', 'composition_thirds', 'composition_balance', 'composition_symmetry',
            'color_saturation', 'color_brightness', 'color_warmth', 'color_contrast', 'color_clarity',
            'fg_color_diff', 'fg_area_diff', 'fg_texture_diff', 'fg_text_legibility'
        ]
        for k in all_dims:
            vals = []
            for r in reports:
                v = getattr(r, k, 0)
                if v is None: v = 0
                # 归一化逻辑，方便可视化
                if k in ['fg_color_diff', 'composition_symmetry', 'fg_text_legibility', 'composition_diagonal', 'composition_thirds', 'composition_balance']:
                    vals.append(v)
                else:
                    vals.append(v * 100)
            dist_data[k] = vals
        return dist_data

    def train(self, pos_files, neg_files, config, auto_weight_enable=True):
        """
        对外接口：双向训练
        返回: (final_profile, dist_data, stats)
        """
        # 1. 分析正向样本
        reps_pos = self._process_images(pos_files, config)
        if not reps_pos: raise ValueError("没有有效的正向图片")
            
        prof_pos = self.manager.create_profile(reps_pos)
        
        # 2. 分析负向样本
        prof_neg = {}
        reps_neg = []
        if neg_files:
            reps_neg = self._process_images(neg_files, config)
            if reps_neg: prof_neg = self.manager.create_profile(reps_neg)
        
        # 3. 处理权重
        final_weights = config.get('weights', {}).copy()
        if auto_weight_enable and len(reps_pos) > 1:
            auto_weights = self._calculate_auto_weights(reps_pos)
            final_weights.update(auto_weights)

        # 4. 【核心修复】提取正向样本的分布数据 (用于前端画图)
        dist_data = self._extract_distribution_data(reps_pos)

        # 5. 提取计算出的容差 (Tolerance)
        # BenchmarkManager 已经在 prof_pos 中计算好了 tolerance，我们需要显式保留它
        # 这里的 prof_pos 结构是 { 'dim_key': {'target': x, 'tolerance': y}, ... }
        final_tolerances = {}
        for k, v in prof_pos.items():
            if isinstance(v, dict) and 'tolerance' in v:
                final_tolerances[k] = v['tolerance']
        
        # 如果有传入的 config 覆盖，以 config 为准吗？不，训练时应该以训练出的容差为准，除非人工锁死
        # 这里我们优先使用训练出的容差，如果训练不出来，才用默认的
        
        # 6. Luv 分布曲线
        luv_engine = LUVAnalysisEngine()
        avg_luv_dist = {'dist_L': np.zeros(100), 'dist_C': np.zeros(100), 'dist_H': np.zeros(100)}
        valid_img_count = 0
        for f in pos_files:
            try:
                if hasattr(f, 'seek'):
                    f.seek(0)
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                dists = luv_engine.extract_luv_distributions(img)
                avg_luv_dist['dist_L'] += dists['dist_L']
                avg_luv_dist['dist_C'] += dists['dist_C']
                avg_luv_dist['dist_H'] += dists['dist_H']
                valid_img_count += 1
            except Exception:
                continue
        if valid_img_count > 0:
            avg_luv_dist['dist_L'] /= valid_img_count
            avg_luv_dist['dist_C'] /= valid_img_count
            avg_luv_dist['dist_H'] /= valid_img_count

        # 7. 组装最终 Profile
        final_profile = {
            "positive": prof_pos,
            "negative": prof_neg,
            "weights": final_weights,
            "tolerances": final_tolerances,
            "luv_curves": avg_luv_dist
        }
        
        stats = {
            "pos_count": len(reps_pos),
            "neg_count": len(reps_neg),
            "auto_weighted": auto_weight_enable
        }
        
        return final_profile, dist_data, stats