## 问题定位
- 报错：name 'score_symmetry' is not defined。发生在返回 OmniReport 时使用了未定义变量。
- 可疑位置：OmniVisualEngine.analyze 中对称性计算块；需要初始化默认值并保证任何异常下仍有值。

## 修复方案
- 在对称性分析前统一初始化：`score_symmetry = 0.0`、`vis_symmetry_heatmap = None`。
- 将计算逻辑置于 try/except 块内，异常时保留默认值。
- 在返回 OmniReport 前确保所有使用的变量均已定义：对角线/三分法/平衡/对称性等。
- 运行编译检查并修复潜在同类问题（如 `vis_edge_fg/vis_edge_bg` 等变量均有默认值）。

## 验证与重启
- 本地编译：`py_compile fenxi/omni_engine.py fenxi/app.py`。
- 停止当前 Streamlit 实例并在端口 8510 重启；打开 `http://localhost:8510/` 验证单图诊断无报错。
- 使用包含明显左右对称元素的图片，检查“对称热力”显示与分数输出稳定。

请确认，我将立即实施修复并重启服务。