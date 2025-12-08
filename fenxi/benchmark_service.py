import cv2
import numpy as np
import pickle
import base64
from omni_engine import OmniVisualEngine, BenchmarkManager

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ 未检测到 scikit-learn。流形学习 (Manifold Learning) 功能将不可用。")

class AestheticManifold:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.is_fitted = False
        self.vectors = None
        self.filenames = []
        self.pca_coords = None
        self.input_dim = 0 
        
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=2)
            self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        else:
            self.scaler = None; self.pca = None; self.knn = None

    def fit(self, vectors, filenames):
        if not SKLEARN_AVAILABLE or not vectors: return
        
        # Record input dimension for normalization
        self.input_dim = len(vectors[0]) 
        
        if len(vectors) < self.n_neighbors:
            self.n_neighbors = max(1, len(vectors))
            self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        
        self.vectors = np.array(vectors)
        self.filenames = filenames
        
        try:
            self.vectors_norm = self.scaler.fit_transform(self.vectors)
            self.knn.fit(self.vectors_norm)
            if len(vectors) > 2:
                self.pca_coords = self.pca.fit_transform(self.vectors_norm)
            else:
                self.pca_coords = np.zeros((len(vectors), 2))
            self.is_fitted = True
        except Exception as e:
            print(f"Manifold fit error: {e}")
            self.is_fitted = False

    def evaluate(self, target_vector):
        if not self.is_fitted or not SKLEARN_AVAILABLE: return 0.0, [], (0, 0)
        
        if len(target_vector) != self.input_dim:
            print(f"⚠️ Manifold dimension mismatch: Model expects {self.input_dim}, got {len(target_vector)}")
            return 0.0, ["维度不匹配"], (0, 0)

        try:
            target_norm = self.scaler.transform([target_vector])
            distances, indices = self.knn.kneighbors(target_norm)
            
            # [Fix] Distance Normalization
            raw_dist = np.mean(distances)
            dim_factor = np.sqrt(self.input_dim) if self.input_dim > 0 else 1.0
            norm_dist = raw_dist / dim_factor
            
            # Map normalized distance to 0-100 score
            score = 100 * np.exp(-0.5 * (norm_dist ** 2))
            
            if self.pca: vis_coord = self.pca.transform(target_norm)[0]
            else: vis_coord = (0, 0)
                
            neighbor_names = [self.filenames[i] for i in indices[0]]
            return score, neighbor_names, vis_coord
        except Exception as e: 
            print(f"Eval error: {e}")
            return 0.0, [], (0, 0)

    def get_visualization_data(self):
        if not self.is_fitted: return None
        return { "x": self.pca_coords[:, 0].tolist(), "y": self.pca_coords[:, 1].tolist(), "filenames": self.filenames }
    
    def save(self): return pickle.dumps(self)
    
    @staticmethod
    def load(data):
        try: return pickle.loads(data)
        except Exception: return None

class BenchmarkTrainer:
    def __init__(self):
        self.engine = OmniVisualEngine()
        self.manager = BenchmarkManager()

    def _process_images(self, file_buffers, config):
        reports = []
        vectors = []
        filenames = []
        
        for f in file_buffers:
            if hasattr(f, 'seek'): f.seek(0)
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                rep = self.engine.analyze(img, config=config)
                reports.append(rep)
                if hasattr(rep, 'to_feature_vector'):
                    vectors.append(rep.to_feature_vector())
                else:
                    # Fallback should match new 18 dim size
                    vectors.append(np.zeros(18)) 
                filenames.append(f.name)
                
        return reports, vectors, filenames

    def calculate_hybrid_score(self, report, manifold, profile):
        # 1. Manifold Score
        score_manifold = 0.0
        neighbors = []
        
        if manifold and manifold.is_fitted:
            try:
                vec = report.to_feature_vector()
                score_manifold, neighbors, _ = manifold.evaluate(vec)
            except Exception as e:
                print(f"Manifold eval failed: {e}")
                score_manifold = 0.0
        
        # 2. Rule Score
        res_rules = self.manager.score_against_benchmark(report, profile)
        score_rules = res_rules['total_score']
        
        # 3. Penalty
        penalty_factor = 0.0
        penalty_reasons = []
        
        if report.color_clarity < 0.5:
            penalty_factor += 0.2; penalty_reasons.append("画质模糊 (-20%)")
        if report.fg_text_present and report.fg_text_legibility < 40:
            penalty_factor += 0.15; penalty_reasons.append("文字难辨 (-15%)")
        if report.comp_balance_score < 30:
            penalty_factor += 0.1; penalty_reasons.append("构图失衡 (-10%)")
            
        penalty_factor = min(0.5, penalty_factor)
        multiplier = 1.0 - penalty_factor
        
        # 4. Fusion
        if score_manifold == 0.0: alpha = 0.0
        else: alpha = 0.6
            
        raw_score = (alpha * score_manifold) + ((1 - alpha) * score_rules)
        final_score = raw_score * multiplier
        
        if final_score >= 90: rating = "S (卓越)"
        elif final_score >= 80: rating = "A (优秀)"
        elif final_score >= 70: rating = "B (良好)"
        elif final_score >= 60: rating = "C (合格)"
        else: rating = "D (不合格)"
        
        return {
            "total_score": round(final_score, 1),
            "rating_level": rating,
            "components": {
                "manifold_score": round(score_manifold, 1),
                "rule_score": round(score_rules, 1),
                "penalty_factor": penalty_factor,
                "penalty_reasons": penalty_reasons,
                "neighbors": neighbors
            },
            "rule_details": res_rules['details']
        }

    def _calculate_auto_weights(self, reports_pos, reports_neg=None):
        # [Cleaned] Removed 'fg_text_contrast' - Total 18 dims
        all_dims = [
            'comp_balance_score', 'comp_layout_score', 'comp_negative_space_score', 
            'comp_visual_flow_score', 'comp_visual_order_score',
            'color_saturation', 'color_brightness', 'color_warmth', 'color_contrast', 'color_clarity', 'color_harmony',
            'fg_color_diff', 'fg_area_diff', 'fg_texture_diff', 'fg_text_legibility',
            'text_alignment_score', 'text_hierarchy_score', 'text_content_ratio'
        ]
        suggested_weights = {}
        def get_values(reports, key):
            vals = []
            for r in reports:
                val = getattr(r, key, 0)
                if key in ['fg_text_legibility'] and not getattr(r, 'fg_text_present', False): continue
                if val is not None: vals.append(float(val))
            return vals

        use_fdr = reports_neg is not None and len(reports_neg) >= 2 and len(reports_pos) >= 2
        for k in all_dims:
            vals_pos = get_values(reports_pos, k)
            if not vals_pos or len(vals_pos) < 2:
                suggested_weights[k] = 1.0; continue
            
            weight = 1.0
            if use_fdr:
                vals_neg = get_values(reports_neg, k)
                if not vals_neg or len(vals_neg) < 2: use_fallback = True
                else:
                    use_fallback = False
                    mu_pos = np.mean(vals_pos); mu_neg = np.mean(vals_neg)
                    var_pos = np.var(vals_pos); var_neg = np.var(vals_neg)
                    fdr = ((mu_pos - mu_neg) ** 2) / (var_pos + var_neg + 1e-6)
                    weight = max(0.5, min(6.0, 0.5 + fdr))
            else: use_fallback = True
            
            if use_fallback:
                std_dev = np.std(vals_pos)
                scale = 100.0 if np.max(vals_pos) > 1.5 else 1.0
                weight = max(0.5, min(4.0, 0.2 / ((std_dev/scale) + 0.05)))
            
            suggested_weights[k] = round(weight, 1)
        return suggested_weights

    def _extract_distribution_data(self, reports):
        dist_data = {}
        # [Cleaned] Removed 'fg_text_contrast' - Total 18 dims
        all_dims = [
            'comp_balance_score', 'comp_layout_score', 'comp_negative_space_score', 
            'comp_visual_flow_score', 'comp_visual_order_score',
            'color_saturation', 'color_brightness', 'color_warmth', 'color_contrast', 'color_clarity', 'color_harmony',
            'fg_color_diff', 'fg_area_diff', 'fg_texture_diff', 'fg_text_legibility',
            'text_alignment_score', 'text_hierarchy_score', 'text_content_ratio'
        ]
        for k in all_dims:
            vals = []
            for r in reports:
                v = getattr(r, k, 0)
                if v is None: v = 0
                
                # Logic copied from OmniEngine logic
                if k in ['comp_balance_score', 'comp_layout_score', 'comp_negative_space_score', 
                         'comp_visual_flow_score', 'comp_visual_order_score', 
                         'color_harmony', 'fg_text_legibility', 'fg_color_diff',
                         'text_alignment_score', 'text_hierarchy_score', 'text_content_ratio', 'fg_texture_diff']:
                     vals.append(v)
                elif k in ['color_warmth', 'color_saturation', 'color_brightness', 'color_clarity', 'fg_area_diff']:
                    vals.append(v * 100.0)
                elif k == 'color_contrast':
                    vals.append(min(100.0, (v / 0.3) * 100.0))
                else:
                    vals.append(v)
            dist_data[k] = vals
        return dist_data

    def train(self, pos_files, neg_files, config, auto_weight_enable=True):
        reps_pos, vecs_pos, names_pos = self._process_images(pos_files, config)
        if not reps_pos: raise ValueError("没有有效的正向图片")
        
        manifold = AestheticManifold()
        if SKLEARN_AVAILABLE and vecs_pos: manifold.fit(vecs_pos, names_pos)
            
        reps_neg, _, _ = self._process_images(neg_files, config) if neg_files else ([], [], [])
        
        prof_pos = self.manager.create_profile(reps_pos)
        prof_neg = self.manager.create_profile(reps_neg) if reps_neg else {}
        
        final_weights = config.get('weights', {}).copy()
        if auto_weight_enable:
            auto_weights = self._calculate_auto_weights(reps_pos, reps_neg if reps_neg else None)
            final_weights.update(auto_weights)

        dist_data_pos = self._extract_distribution_data(reps_pos)
        dist_data_neg = self._extract_distribution_data(reps_neg) if reps_neg else {}

        final_tolerances = {}
        for k, v in prof_pos.items():
            if isinstance(v, dict) and 'std' in v: final_tolerances[k] = v['std']
            
        final_profile = {
            "positive": prof_pos,
            "negative": prof_neg,
            "weights": final_weights,
            "manifold_bytes": base64.b64encode(manifold.save()).decode('utf-8') if manifold.is_fitted else ""
        }
        
        stats = {
            "pos_count": len(reps_pos),
            "neg_count": len(reps_neg),
            "auto_weighted": auto_weight_enable
        }
        
        return final_profile, {"pos": dist_data_pos, "neg": dist_data_neg}, stats, manifold