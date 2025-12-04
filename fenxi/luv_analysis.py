import cv2
import numpy as np
from scipy.stats import wasserstein_distance

class LUVAnalysisEngine:
    def __init__(self, bins=100):
        self.bins = bins
        self.bin_centers = np.linspace(0, 1, bins)

    def extract_luv_distributions(self, img_bgr):
        img_float = img_bgr.astype(np.float32) / 255.0
        luv = cv2.cvtColor(img_float, cv2.COLOR_BGR2Luv)
        L, u, v = cv2.split(luv)
        norm_L = L / 100.0
        C = np.sqrt(u**2 + v**2)
        norm_C = np.clip(C / 150.0, 0, 1)
        H = np.arctan2(v, u)
        norm_H = (H + np.pi) / (2 * np.pi)
        hist_L, _ = np.histogram(norm_L, bins=self.bins, range=(0, 1), density=True)
        hist_C, _ = np.histogram(norm_C, bins=self.bins, range=(0, 1), density=True)
        hist_H, _ = np.histogram(norm_H, bins=self.bins, range=(0, 1), density=True)
        return {
            "dist_L": hist_L,
            "dist_C": hist_C,
            "dist_H": hist_H
        }

    def compute_emd_score(self, dist_a, dist_b):
        d_L = wasserstein_distance(self.bin_centers, self.bin_centers, u_weights=dist_a['dist_L'], v_weights=dist_b['dist_L'])
        d_C = wasserstein_distance(self.bin_centers, self.bin_centers, u_weights=dist_a['dist_C'], v_weights=dist_b['dist_C'])
        d_H = wasserstein_distance(self.bin_centers, self.bin_centers, u_weights=dist_a['dist_H'], v_weights=dist_b['dist_H'])
        total_diff = (d_L * 0.4) + (d_C * 0.4) + (d_H * 0.2)
        score = max(0, 100 - (total_diff * 500))
        return score, {"diff_L": d_L, "diff_C": d_C, "diff_H": d_H}