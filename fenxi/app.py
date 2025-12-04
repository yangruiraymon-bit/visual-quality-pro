import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import pandas as pd
import io
import zipfile
import time
# 确保 omni_engine.py 在同一目录下
from omni_engine import OmniVisualEngine, AestheticDiagnostician

# === 1. 页面基础配置 ===
st.set_page_config(page_title="全能视觉分析 Pro", layout="wide", page_icon="🧿")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 5rem; }
        h1 { font-size: 2.0rem !important; margin-bottom: 0.5rem !important; }
        section[data-testid="stSidebar"] { background-color: #f8f9fa; }
        .stButton button { width: 100%; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# === 2. 状态初始化 ===
if 'batch_df' not in st.session_state: st.session_state.batch_df = None
if 'batch_zip' not in st.session_state: st.session_state.batch_zip = None
if 'batch_imgs' not in st.session_state: st.session_state.batch_imgs = [] # 存储用于Excel的图片流字典
if 'batch_logs' not in st.session_state: st.session_state.batch_logs = []
if 'processing' not in st.session_state: st.session_state.processing = False

# 初始化引擎
@st.cache_resource
def get_engine():
    return OmniVisualEngine()

engine = get_engine()

# === 3. Excel 生成函数 (核心升级) ===
def to_excel_with_all_images(df, img_dicts):
    """
    将数据和所有对应的可视化图写入 Excel
    df: 数据 DataFrame
    img_dicts: list of dict, 每个元素是 {'diag': bytes, 'thirds': bytes...}
    """
    output = io.BytesIO()
    
    # 定义图片列的顺序和标题
    img_columns = [
        ('vis_diag', '构图:对角线'),
        ('vis_thirds', '构图:三分法'),
        ('vis_balance', '构图:平衡'),
        ('vis_symmetry_heatmap', '构图:对称热力'),
        ('vis_warmth', '色彩:暖色分布'),
        ('vis_saturation', '色彩:饱和度'),
        ('vis_brightness', '色彩:亮度'),
        ('vis_contrast', '色彩:对比度'),
        ('vis_clarity', '色彩:清晰度'),
        ('vis_mask', '图底:面积Mask'),
        ('vis_color_contrast', '图底:色彩抽离'),
        ('vis_edge_composite', '图底:纹理对比'),
        ('vis_text_analysis', '文字:易读性')
    ]

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='分析结果', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['分析结果']
        
        # 基础样式
        header_fmt = workbook.add_format({'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'bg_color': '#DDEBF7'})
        base_col_count = len(df.columns)
        
        # 设置行高 (适应图片高度 100px -> 约75磅)
        worksheet.set_default_row(75)
        
        # 写入图片列的表头
        for i, (key, title) in enumerate(img_columns):
            col_idx = base_col_count + i
            worksheet.write(0, col_idx, title, header_fmt)
            worksheet.set_column(col_idx, col_idx, 18) # 设置列宽
            
        # 遍历每一行数据
        for row_idx, img_dict in enumerate(img_dicts):
            # Excel 行索引从 1 开始 (0是表头)
            excel_row = row_idx + 1
            
            if img_dict is None: continue
            
            for i, (key, title) in enumerate(img_columns):
                img_data = img_dict.get(key)
                if img_data:
                    col_idx = base_col_count + i
                    # 插入图片
                    worksheet.insert_image(excel_row, col_idx, f"{key}.png", {
                        'image_data': img_data,
                        'x_scale': 1, 'y_scale': 1, # 图片已经在预处理时缩放好了
                        'object_position': 1 # 居中
                    })
                    
    return output.getvalue()

# === 4. 批量处理逻辑 (Callback) ===
def run_batch_process(files, cfg, need_csv, need_zip):
    st.session_state.processing = True
    st.session_state.batch_logs = []
    
    rows = []
    img_dicts_list = [] # 用于存储 Excel 图片流
    
    zip_buffer = io.BytesIO() if need_zip else None
    zf = zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) if need_zip else None
    
    total = len(files)
    progress_bar = st.progress(0)
    
    for idx, f in enumerate(files):
        log_msg = f"[{idx+1}/{total}] 处理中: {f.name}..."
        st.session_state.batch_logs.append(log_msg)
        
        try:
            f_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                st.session_state.batch_logs.append(f"❌ 错误: 无法读取 {f.name}")
                rows.append({"文件名": f.name, "总分": 0, "评级": "Error"})
                img_dicts_list.append(None)
                continue

            # 分析
            d = engine.analyze(img_bgr, config=cfg)
            rep = AestheticDiagnostician.generate_report(d, config=cfg)
            
            # 数据行
            probs = []
            if d.color_clarity < 0.1: probs.append("雾感重")
            if d.fg_color_diff < 40: probs.append("主体不清")
            if getattr(d,'fg_text_present',False) and d.fg_text_legibility < 60: probs.append("文字难辨")
            
            row = {
                "文件名": f.name,
                "总分": rep['total_score'],
                "评级": rep['rating_level'],
                "风格": " ".join(rep['style_tags']),
                "问题摘要": "、".join(probs) if probs else "无",
                "构图_对角线": d.composition_diagonal,
                "构图_三分法": d.composition_thirds,
                "构图_平衡度": d.composition_balance,
                "构图_对称性": getattr(d, 'composition_symmetry', 0),
                "色彩_暖色占比": d.color_warmth,
                "色彩_饱和度": d.color_saturation,
                "色彩_亮度": d.color_brightness,
                "色彩_对比度": d.color_contrast,
                "色彩_清晰度": d.color_clarity,
                "图底_面积差": d.fg_area_diff,
                "图底_色差": d.fg_color_diff,
                "图底_纹理差": d.fg_texture_diff,
                "文字_易读性": getattr(d, 'fg_text_legibility', 0)
            }
            rows.append(row)
            
            # === 处理图片 ===
            # 1. 准备 Excel 用的缩略图 (存入内存)
            # 2. 准备 ZIP 用的原图 (写入 ZIP)
            
            current_imgs = {} # 存放当前行的所有缩略图流
            
            # 需要保存的所有字段名
            keys = [
                'vis_diag', 'vis_thirds', 'vis_balance', 'vis_symmetry_heatmap',
                'vis_warmth', 'vis_saturation', 'vis_brightness', 'vis_contrast', 'vis_clarity',
                'vis_mask', 'vis_color_contrast', 'vis_edge_composite', 'vis_text_analysis'
            ]
            
            base_name = f.name.rsplit('.', 1)[0]
            
            for key in keys:
                img_arr = getattr(d, key, None)
                if img_arr is not None:
                    # 统一转 PIL RGB
                    if len(img_arr.shape)==3 and img_arr.shape[2]==3: 
                        pil_img = Image.fromarray(img_arr)
                    else:
                        pil_img = Image.fromarray(img_arr)
                    
                    # A. 为 Excel 制作缩略图 (高度固定 100px)
                    # 保持比例缩放
                    w_orig, h_orig = pil_img.size
                    ratio = 100.0 / h_orig
                    new_w = int(w_orig * ratio)
                    thumb = pil_img.resize((new_w, 100))
                    
                    b_thumb = io.BytesIO()
                    thumb.save(b_thumb, format='PNG')
                    current_imgs[key] = b_thumb
                    
                    # B. 为 ZIP 保存高清图
                    if need_zip and zf:
                        b_full = io.BytesIO()
                        pil_img.save(b_full, format='JPEG', quality=85)
                        zf.writestr(f"diagnostics/{base_name}_{key}.jpg", b_full.getvalue())
            
            img_dicts_list.append(current_imgs)

        except Exception as e:
            st.session_state.batch_logs.append(f"❌ 异常: {f.name} - {str(e)}")
            rows.append({"文件名": f.name, "总分": 0, "评级": "Error", "问题摘要": str(e)})
            img_dicts_list.append(None)
        
        progress_bar.progress((idx + 1) / total)

    if zf: zf.close()
    
    # 更新 Session State
    st.session_state.batch_df = pd.DataFrame(rows)
    st.session_state.batch_imgs = img_dicts_list # 保存图片流列表
    st.session_state.batch_zip = zip_buffer.getvalue() if need_zip else None
    st.session_state.processing = False
    st.session_state.batch_logs.append("✅ 所有任务处理完成！")


# ==========================================
# 🟢 侧边栏布局
# ==========================================
with st.sidebar:
    st.header("🧿 视觉分析台")
    mode = st.radio("工作模式", ["单图诊断", "批量工厂"], index=0)
    st.divider()
    
    with st.expander("⚙️ 算法参数配置", expanded=False):
        p_width = st.slider("分析分辨率", 256, 1024, 512, 128)
        k_num = st.slider("聚类数", 2, 8, 5)
        st.caption("构图"); t_diag = st.slider("对角线容差", 0.1, 0.5, 0.3)
        t_sym_blur = st.slider("对称模糊K", 1, 51, 31, 2)
        st.caption("图底"); ref_tex = st.slider("纹理基准", 10.0, 100.0, 50.0)
    with st.expander("⚖️ 评分权重定制 (0=不计分)", expanded=False):
        st.caption("📐 构图维度")
        wc1 = st.slider("对角线", 0.0, 5.0, 1.0, 0.1, key="w_c1")
        wc2 = st.slider("三分法", 0.0, 5.0, 1.0, 0.1, key="w_c2")
        wc3 = st.slider("平衡度", 0.0, 5.0, 1.0, 0.1, key="w_c3")
        wc4 = st.slider("稳定性", 0.0, 5.0, 1.0, 0.1, key="w_c4")
        st.caption("🎨 色彩维度")
        wl1 = st.slider("清晰度", 0.0, 5.0, 2.0, 0.1, key="w_l1")
        wl2 = st.slider("对比度", 0.0, 5.0, 1.0, 0.1, key="w_l2")
        wl3 = st.slider("饱和度", 0.0, 5.0, 1.0, 0.1, key="w_l3")
        wl4 = st.slider("暖色调", 0.0, 5.0, 0.5, 0.1, key="w_l4", help="商业/美食摄影建议调高此权重 (暖色=高分)")
        wl5 = st.slider("亮度", 0.0, 5.0, 0.5, 0.1, key="w_l5", help="商业/美食摄影建议调高 (0.45-0.75区间得满分)")
        st.caption("🌗 图底维度")
        wf1 = st.slider("主体色差", 0.0, 5.0, 1.5, 0.1, key="w_f1")
        wf2 = st.slider("面积差异", 0.0, 5.0, 1.0, 0.1, key="w_f2")
        wf3 = st.slider("纹理差异", 0.0, 5.0, 0.5, 0.1, key="w_f3")
        wf4 = st.slider("文字易读", 0.0, 5.0, 2.0, 0.1, key="w_f4")
        
    config = {
        'process_width': p_width, 'seg_kmeans_k': k_num, 'comp_diag_slope': t_diag, 
        'comp_sym_blur_k': t_sym_blur, 'fg_tex_norm': ref_tex, 
        'comp_thirds_slope': 0.2, 'comp_sym_tolerance': 120.0, 
        'color_clarity_thresh': 0.7, 'text_score_thresh': 60.0,
        'w_comp_diagonal': wc1, 'w_comp_thirds': wc2, 'w_comp_balance': wc3, 'w_comp_symmetry': wc4,
        'w_color_clarity': wl1, 'w_color_contrast': wl2, 'w_color_saturation': wl3, 'w_color_warmth': wl4, 'w_color_brightness': wl5,
        'w_fg_color': wf1, 'w_fg_area': wf2, 'w_fg_texture': wf3, 'w_fg_text': wf4
    }

    if mode == "批量工厂":
        st.subheader("📂 批量任务")
        batch_files = st.file_uploader("多选图片", type=["jpg","png"], accept_multiple_files=True)
        c1, c2 = st.columns(2)
        with c1: opt_csv = st.checkbox("数据表", value=True)
        with c2: opt_zip = st.checkbox("图包", value=True)
        
        if batch_files:
            st.button("🚀 开始运行", type="primary", on_click=run_batch_process, args=(batch_files, config, opt_csv, opt_zip))
        
        # --- 下载区域 ---
        if st.session_state.batch_df is not None:
            st.divider()
            st.success(f"已生成 {len(st.session_state.batch_df)} 条记录")
            
            # 1. 简单的 CSV 下载
            st.download_button("📄 下载纯数据 (.csv)", 
                               data=st.session_state.batch_df.to_csv(index=False).encode('utf-8-sig'), 
                               file_name="batch_data.csv", mime="text/csv")
            
            # 2. [核心功能] 带图的 Excel 下载
            if st.session_state.batch_imgs:
                # 实时生成 Excel (因为 BytesIO 对象是一次性的，建议点击时生成)
                excel_data = to_excel_with_all_images(st.session_state.batch_df, st.session_state.batch_imgs)
                st.download_button("📊 下载全景报表 (.xlsx)", 
                                   data=excel_data, 
                                   file_name="visual_report_full.xlsx", 
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   type="primary",
                                   help="包含数据和所有13个维度的缩略图，文件较大")
            
            # 3. ZIP 下载
            if st.session_state.batch_zip:
                st.download_button("📦 下载原图包 (.zip)", 
                                   data=st.session_state.batch_zip, 
                                   file_name="batch_images.zip", mime="application/zip")

# ==========================================
# 🔵 主界面逻辑 (保持不变，或根据需要简化)
# ==========================================
if mode == "批量工厂":
    st.title("📦 批量处理中心")
    if st.session_state.processing:
        st.info("正在后台处理中，请勿刷新页面...")
        with st.expander("查看实时日志", expanded=True):
            for log in st.session_state.batch_logs[-5:]: st.text(log)
    
    if st.session_state.batch_df is not None:
        st.subheader("📊 结果预览")
        st.dataframe(st.session_state.batch_df.style.background_gradient(subset=['总分'], cmap="RdYlGn"), use_container_width=True, height=600)
    else:
        st.info("👈 请在左侧侧边栏上传图片并点击【开始运行】")

elif mode == "单图诊断":
    st.title("🧿 单图深度诊断")
    uploaded_file = st.file_uploader("上传单张图片", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        img_np = np.array(image_pil.convert('RGB'))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        with st.spinner("AI 正在全维扫描..."):
            data = engine.analyze(img_bgr, config=config)
            report = AestheticDiagnostician.generate_report(data, config=config)
        st.header("📝 AI 美学诊断报告")
        rep_c1, rep_c2 = st.columns([1, 2])
        with rep_c1:
            st.metric("综合美学评分", f"{report['total_score']} / 100", report['rating_level'])
        with rep_c2:
            st.caption("AI 识别风格标签：")
            tags_html = "".join([f"<span style='background-color:#eee; padding:4px 10px; margin:0 5px; border-radius:15px; font-size:14px'>{tag}</span>" for tag in report['style_tags']])
            st.markdown(tags_html, unsafe_allow_html=True)
            st.info(f"💡 **AI 总结**：{report['summary']}")
        adv_c1, adv_c2 = st.columns(2)
        with adv_c1:
            st.subheader("✅ 亮点 (Pros)")
            if report['pros']:
                for item in report['pros']:
                    st.markdown(f"- {item}")
            else:
                st.write("暂无显著亮点，表现平稳。")
        with adv_c2:
            st.subheader("⚠️ 改进点 (Cons)")
            if report['cons']:
                for item in report['cons']:
                    st.markdown(f"- {item}")
            else:
                st.write("未发现明显缺陷，非常完美！")
        st.divider()
        top_c1, top_c2 = st.columns([1, 1])
        with top_c1:
            st.subheader("原始图像")
            st.image(image_pil, use_container_width=True)
        with top_c2:
            st.subheader("特征雷达图")
            categories = [
                '<b>构图</b><br>对角线', '<b>构图</b><br>三分法', '<b>构图</b><br>平衡', '<b>构图</b><br>稳定性',
                '<b>色彩</b><br>暖色', '<b>色彩</b><br>饱和度', '<b>色彩</b><br>亮度', '<b>色彩</b><br>对比度', '<b>色彩</b><br>清晰度',
                '<b>图底</b><br>面积差', '<b>图底</b><br>色差', '<b>图底</b><br>纹理',
                '<b>文字</b><br>易读性'
            ]
            v1 = float(getattr(data, 'composition_diagonal', 0))
            v2 = float(getattr(data, 'composition_thirds', 0))
            v3 = float(getattr(data, 'composition_balance', 0))
            v4 = float(getattr(data, 'composition_symmetry', getattr(data, 'color_symmetry', 0)))
            v5 = float(getattr(data, 'color_warmth', 0)) * 100.0
            v6 = float(getattr(data, 'color_saturation', 0)) * 100.0
            v7 = float(getattr(data, 'color_brightness', 0)) * 100.0
            raw_contrast = float(getattr(data, 'color_contrast', 0))
            v8 = min(100.0, (raw_contrast / 0.3) * 100.0)
            v9 = float(getattr(data, 'color_clarity', 0)) * 100.0
            v10 = float(getattr(data, 'fg_area_diff', 0)) * 100.0
            raw_color_diff = float(getattr(data, 'fg_color_diff', 0))
            v11 = min(100.0, raw_color_diff)
            v12 = float(getattr(data, 'fg_texture_diff', 0)) * 100.0
            v13 = float(getattr(data, 'fg_text_legibility', 0)) if getattr(data, 'fg_text_present', False) else 0.0
            values = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13]
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill='toself',
                fillcolor='rgba(0, 191, 255, 0.2)',
                line=dict(color='deepskyblue', width=2),
                mode='lines+markers',
                marker=dict(size=6, color='dodgerblue', symbol='circle'),
                hoverinfo='text',
                text=[f"{c.replace('<br>', ' ')}: {v:.1f}" for c, v in zip(categories_closed, values_closed)]
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickfont=dict(size=9, color='gray'),
                        tickvals=[20, 60, 100],
                        gridcolor='rgba(0,0,0,0.1)',
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=11, color='#333'),
                        rotation=90,
                        direction="clockwise"
                    )
                ),
                showlegend=False,
                margin=dict(l=50, r=50, t=30, b=30),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'staticPlot': False})
        st.divider()
        tab1, tab2, tab3 = st.tabs(["📐 构图", "🎨 色彩", "🌗 图底 & 文字"])
        with tab1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("对角线主导 (范围: 0-100)", f"{data.composition_diagonal}")
            c2.metric("三分法契合 (范围: 0-100)", f"{data.composition_thirds}")
            c3.metric("物理分布平衡 (范围: 0-100)", f"{data.composition_balance}")
            sym_score = float(getattr(data, 'composition_symmetry', getattr(data, 'color_symmetry', 0)))
            c4.metric("视觉色彩平衡 (范围: 0-100)", f"{sym_score}")
            img_c1, img_c2, img_c3, img_c4 = st.columns(4)
            if getattr(data, 'vis_diag', None) is not None: img_c1.image(data.vis_diag, use_container_width=True, caption="对角线辅助线")
            if getattr(data, 'vis_thirds', None) is not None: img_c2.image(data.vis_thirds, use_container_width=True, caption="三分法连线")
            if getattr(data, 'vis_balance', None) is not None: img_c3.image(data.vis_balance, use_container_width=True, caption="左右力矩分布")
            if getattr(data, 'vis_symmetry_heatmap', None) is not None: img_c4.image(data.vis_symmetry_heatmap, use_container_width=True, caption="镜像色差热力图")
        with tab2:
            st.markdown("#### 🎨 色彩与光影分析")
            c1, c2 = st.columns([1.5, 1])
            with c1:
                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    st.metric("暖色调占比 (范围: 0-100%)", f"{data.color_warmth:.0%}")
                    if getattr(data, 'vis_warmth', None) is not None: st.image(data.vis_warmth, use_container_width=True, caption="分布图(红暖蓝冷)")
                with r1c2:
                    st.metric("平均饱和度 (范围: 0.0-1.0 Chroma)", f"{data.color_saturation:.2f}")
                    if getattr(data, 'vis_saturation', None) is not None: st.image(data.vis_saturation, use_container_width=True, caption="热力图(红高蓝低)")
                st.divider()
                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    st.metric("平均亮度 (范围: 0-1 L)", f"{data.color_brightness:.2f}")
                    if getattr(data, 'vis_brightness', None) is not None: st.image(data.vis_brightness, use_container_width=True, caption="亮度分布(人眼灰度)")
                with r2c2:
                    st.metric("光影对比度 (范围: 0.0-0.5+ StdDev)", f"{data.color_contrast:.2f}")
                    if getattr(data, 'vis_contrast', None) is not None: st.image(data.vis_contrast, use_container_width=True, caption="明暗色阶(黑/灰/白)")
            with c2:
                st.metric("高亮区域占比 (范围: 0-100%)", f"{data.color_clarity:.0%}")
                if getattr(data, 'vis_clarity', None) is not None: st.image(data.vis_clarity, use_container_width=True, caption="清晰度分布图 (聚光灯效果)")
                score_clarity = data.color_clarity
                if score_clarity > 0.85:
                    st.error("💥 严重过曝：高光溢出，画面细节丢失，视觉刺眼。")
                elif score_clarity > 0.3:
                    st.success("☀️ 通透清晰：画面有充足的高光区域，视觉传达效率高。")
                elif score_clarity > 0.1:
                    st.info("☁️ 柔和/自然：光照分布均匀，可能具有电影感或胶片感。")
                else:
                    st.warning("🌫️ 沉闷/雾感：高光缺失，画面可能显得灰暗或对焦不清。")
        with tab3:
            st.markdown("#### 🔤 文字易读性诊断")
            if getattr(data, 'fg_text_present', False):
                c1, c2 = st.columns([1.5, 1])
                with c1:
                    if getattr(data, 'vis_text_analysis', None) is not None: st.image(data.vis_text_analysis, use_container_width=True, caption="易读性诊断 (数字为综合评分)")
                with c2:
                    st.metric("文字综合易读性 (范围: 0-100)", f"{data.fg_text_legibility}/100", delta=("优秀" if data.fg_text_legibility > 80 else "需优化"))
                    st.metric("平均对比度 (范围: 0-200+ ΔE)", f"{data.fg_text_contrast:.1f}")
                    st.info("诊断图例：\n- 🟩 绿框：易读 (Score > 60)\n- 🟥 红框：难辨 (Score < 60)")
                    if data.fg_text_legibility < 60: st.warning("⚠️ 建议：给红框内的文字添加阴影、描边或半透明底板。")
            else:
                st.info("ℹ️ 画面中未检测到明显文字。")
            st.divider()
            st.markdown("#### 🖼️ 图形主体分析")
            c1a, c2a, c3a = st.columns(3)
            c1a.metric("面积差异 (范围: 0.0-1.0)", f"{data.fg_area_diff:.2f}")
            c2a.metric("色彩差异 (范围: 0-200+ ΔE)", f"{data.fg_color_diff:.1f}")
            c3a.metric("纹理差异 (范围: 0.0-1.0)", f"{data.fg_texture_diff:.3f}")
            vc1, vc2, vc3 = st.columns([1, 1, 1.5])
            with vc1:
                st.caption("1. AI 主体分割 (Mask)")
                _mask = getattr(data, 'vis_mask', None)
                if _mask is not None:
                    mask_display = (_mask.astype(np.uint8) * 255) if _mask.max() <= 1 else _mask
                    st.image(mask_display, use_container_width=True, clamp=True)
                else:
                    st.warning("未生成")
            with vc2:
                st.caption("2. 平均色彩抽离 (Color)")
                _color = getattr(data, 'vis_color_contrast', None)
                if _color is not None:
                    st.image(_color, use_container_width=True, caption=f"Diff: {data.fg_color_diff}")
                    if data.fg_color_diff > 100: st.caption("✅ 强对比 (撞色)")
                    elif data.fg_color_diff < 50: st.caption("⚠️ 弱对比 (顺色)")
                else:
                    st.warning("未生成")
            with vc3:
                st.caption("3. 纹理密度对比 (Texture)")
                _comp = getattr(data, 'vis_edge_composite', None)
                if _comp is not None:
                    composite_display = _comp.astype(np.uint8)
                    st.image(composite_display, use_container_width=True, caption="绿=前景 | 红=背景")
                    st.info("🟢 绿色：前景纹理 | 🔴 红色：背景纹理")
                else:
                    st.warning("未生成")