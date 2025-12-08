## 问题概述
- 报错：`OmniVisualEngine` 无 `_analyze_distribution` 方法，但在 `analyze()` 中调用了它。
- 另一个潜在问题：`valid_ocr_items` 在文本评分部分被使用，但未在当前代码中定义。

## 修复方案
- 增加方法：在 `OmniVisualEngine` 中实现 `def _analyze_distribution(self, img_bgr, visual_elements, valid_contours)`。
  - 统计对象中心点、面积、方向角；栅格计数计算空间熵；近邻距离计算间距 CV；面积 CV；方向直方图熵。
  - 输出：`(num_objects, norm_entropy, spacing_cv, size_cv, angle_entropy, vis_dist_entropy, vis_dist_size, vis_dist_angle)`，其中三张可视化图使用现有 `_draw_dist_*` 辅助函数并返回 RGB。
- 修正变量：在 `analyze()` 中构造 `valid_ocr_items`（过滤 `ocr_raw` 中 `prob > 0.3` 的项），替换后续循环的数据源，避免 `NameError`。
- 保持现有返回结构：`OmniReport` 字段 `dist_*` 与 `vis_dist_*` 按现有 UI 使用的命名稳定输出。

## 代码插入位置
- 方法定义：紧随现有 `_draw_dist_*` 辅助方法后，保证依赖存在。
- 变量修正：在 `ocr_raw = self.reader.readtext(img_small)` 之后，生成 `valid_ocr_items` 并用于文本评分循环。

## 验证
- 编译：`py_compile fenxi/omni_engine.py fenxi/app.py`。
- 运行：重启服务，单图诊断加载测试图，确认“🧩 分布”三图与三项指标正常显示；文本评分不再抛错。
- 额外检查：保证 `vis_*` 图均有默认或条件赋值，避免再次出现未定义变量或属性错误。

请确认，我将立即按上述方案补齐方法与变量并重启服务进行验证。