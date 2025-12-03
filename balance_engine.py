import cv2
import numpy as np
from rembg import remove
from dataclasses import dataclass

@dataclass
class BalanceResult:
    score: float          # 距离对称性评分 (0-100)
    left_arm: float       # 左侧力臂长度
    right_arm: float      # 右侧力臂长度
    mass_ratio: float     # 左右质量(面积)比
    equilibrium: float    # 物理力矩平衡度 (0-100)
    visualization: np.ndarray = None

class SymmetryAnalyzer:
    def __init__(self):
        pass

    def analyze(self, image_input: np.ndarray, debug: bool = True) -> BalanceResult:
        h, w = image_input.shape[:2]
        center_x = w // 2
        
        # 1. AI 显著性检测 (获取主体)
        # 转为 RGB 供 rembg 使用
        img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        mask_rgba = remove(img_rgb, alpha_matting=True)
        mask = mask_rgba[:, :, 3] # Alpha 通道
        
        # 二值化
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 2. 分割左右半区
        mask_left = binary_mask[:, :center_x]
        mask_right = binary_mask[:, center_x:]
        
        # 3. 计算左侧指标
        M_L = cv2.moments(mask_left)
        area_left = M_L["m00"]
        
        cx_left_global = 0
        arm_left = 0
        
        if area_left > 0:
            cx_local = M_L["m10"] / area_left
            cy_local = M_L["m01"] / area_left
            cx_left_global = int(cx_local)
            # 左力臂 = 中线 - 左质心
            arm_left = center_x - cx_left_global
            
        # 4. 计算右侧指标
        M_R = cv2.moments(mask_right)
        area_right = M_R["m00"]
        
        cx_right_global = w
        arm_right = 0
        
        if area_right > 0:
            cx_local = M_R["m10"] / area_right
            cy_local = M_R["m01"] / area_right
            # 右质心全局坐标 = 局部坐标 + 半宽
            cx_right_global = int(cx_local + center_x)
            # 右力臂 = 右质心 - 中线
            arm_right = cx_right_global - center_x

        # 5. 计算评分 (核心需求: 距离对称性)
        if arm_left == 0 and arm_right == 0:
            symmetry_score = 0.0 # 无物体
        elif arm_left == 0 or arm_right == 0:
            symmetry_score = 0.0 # 极度不平衡
        else:
            # 计算距离差异率
            diff = abs(arm_left - arm_right)
            max_arm = max(arm_left, arm_right)
            symmetry_score = max(0, 100 * (1 - (diff / max_arm)))

        # 辅助指标: 质量平衡 (力矩平衡)
        # Torque = Force(Area) * Distance
        torque_left = area_left * arm_left
        torque_right = area_right * arm_right
        total_torque = torque_left + torque_right
        equilibrium_score = 0
        if total_torque > 0:
             equilibrium_score = 100 * (1 - abs(torque_left - torque_right) / total_torque)
            
        # 6. 可视化绘制
        vis_img = image_input.copy()
        if debug:
            # 中线
            cv2.line(vis_img, (center_x, 0), (center_x, h), (255, 255, 0), 2) # 青色中线
            
            # 左侧
            if area_left > 0:
                # 质心点
                cv2.circle(vis_img, (cx_left_global, int(h/2)), 10, (0, 255, 0), -1) 
                # 力臂线
                cv2.line(vis_img, (cx_left_global, int(h/2)), (center_x, int(h/2)), (0, 255, 0), 4)
                # 文字
                cv2.putText(vis_img, f"L:{int(arm_left)}px", (cx_left_global-40, int(h/2)-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 右侧
            if area_right > 0:
                # 质心点
                cv2.circle(vis_img, (cx_right_global, int(h/2)), 10, (0, 0, 255), -1)
                # 力臂线
                cv2.line(vis_img, (center_x, int(h/2)), (cx_right_global, int(h/2)), (0, 0, 255), 4)
                # 文字
                cv2.putText(vis_img, f"R:{int(arm_right)}px", (cx_right_global, int(h/2)-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                           
        return BalanceResult(
            score=round(symmetry_score, 1),
            left_arm=round(arm_left, 1),
            right_arm=round(arm_right, 1),
            mass_ratio=round(area_left/area_right if area_right>0 else 0, 2),
            equilibrium=round(equilibrium_score, 1),
            visualization=vis_img
        )