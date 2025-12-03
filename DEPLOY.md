# 部署指南

## Docker 部署
- 构建镜像：`docker build -t omni-engine .`
- 运行容器：`docker run -p 8501:8501 omni-engine`
- 访问：`http://localhost:8501`

## 云平台（示例 Render）
- 新建 Web Service，选择 Docker。
- 暴露端口 `8501`，启动命令无需填写（Dockerfile 已定义）。

## Streamlit Community Cloud
- 准备仓库包含：`fenxi/omni_engine_app.py`、`requirements.txt`。
- 在项目设置中选择入口：`fenxi/omni_engine_app.py`。

## 本地直接运行
- 安装依赖：`python3 -m pip install -r requirements.txt`
- 启动：`python3 -m streamlit run fenxi/omni_engine_app.py`
- 如端口占用：`python3 -m streamlit run fenxi/omni_engine_app.py --server.port 8502`

## 说明
- 已生成 `requirements.txt`，固定当前环境依赖版本。
- Dockerfile 基于 `python:3.9-slim`，安装必要系统库并运行 Streamlit。
- 如使用 `rembg` 推理，若出现后端问题，请安装/升级 `onnxruntime`。