import numpy as np
from scipy.stats import wasserstein_distance

class DistributionScorer:
    def __init__(self):
        self.bins = np.linspace(0, 1, 100)

    def calculate_score(self, curr_hist, bench_hist, sensitivity=400.0):
        if curr_hist is None or bench_hist is None:
            return 0.0
        dist = wasserstein_distance(
            self.bins, self.bins,
            u_weights=curr_hist,
            v_weights=bench_hist
        )
        score = 100 - (dist * sensitivity)
        return max(0.0, min(100.0, score))

    def evaluate_luv_quality(self, curr_dists, bench_dists):
        if not bench_dists:
            return None
        score_l = self.calculate_score(curr_dists['dist_L'], bench_dists['dist_L'], sensitivity=500)
        score_c = self.calculate_score(curr_dists['dist_C'], bench_dists['dist_C'], sensitivity=400)
        score_h = self.calculate_score(curr_dists['dist_H'], bench_dists['dist_H'], sensitivity=300)
        return {
            "score_L": score_l,
            "score_C": score_c,
            "score_H": score_h,
            "avg_score": (score_l * 0.4 + score_c * 0.4 + score_h * 0.2)
        }