# 图片质量量化工具（白盒化诊断）

基于 Streamlit 的全能视觉分析应用：构图/色彩/图底/文字全维诊断，支持可视化与评分报告，含批量导出。

## 本地运行

1. 创建虚拟环境并安装依赖：
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. 启动应用：
   - `python -m streamlit run fenxi/app.py --server.port 8507`

## 目录结构
- `fenxi/app.py`：主界面
- `fenxi/omni_engine.py`：引擎（分割、构图、色彩、图底、文字）
- `.streamlit/config.toml`：运行与主题配置

## 在 Streamlit Community Cloud 发布
1. 将仓库推送到 GitHub（保持上述文件结构）
2. 打开 https://share.streamlit.io ，选择该仓库
3. App file path 填写：`fenxi/app.py`
4. 自动安装 `requirements.txt` 中依赖
5. 启动后即可访问在线应用

> 注意：YOLOv8 分割（`ultralytics`）在 CPU 上可运行，但首次加载会下载模型。若云端资源受限，建议将模型改为更轻量的 `yolov8n-seg.pt`。

## 在容器或云平台部署（可选）
- 使用 `requirements.txt` 安装依赖
- 暴露端口并运行：`python -m streamlit run fenxi/app.py --server.port 8507`
- 如平台支持健康检查，建议检查 8507 端口 HTTP 可用

## 常见问题
- 首次运行慢：模型与 OCR 初始化较慢，耐心等待
- 文字识别效果：`easyocr` 为通用 OCR，复杂排版建议替换更强模型
- 大模型依赖：如资源受限，降低分辨率（Sidebar 或配置）并使用小模型

## 许可
此项目用于演示与评估目的，请在遵守相关开源依赖许可的前提下使用。