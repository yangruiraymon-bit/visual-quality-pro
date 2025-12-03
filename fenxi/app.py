import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import pandas as pd
import io
import zipfile
# 确保 omni_engine.py 在同一目录下，且包含 OmniVisualEngine, AestheticDiagnostician 类
from omni_engine import OmniVisualEngine, AestheticDiagnostician

st.set_page_config(page_title="全能视觉分析 Pro", layout="wide", page_icon="🧿")

# === 侧边栏：参数设置 ===
with st.sidebar:
    st.header("⚙️ 参数配置")
    with st.expander("🛠️ 预处理 & 分割", expanded=True):
        p_width = st.slider("分析分辨率 (Width)", 256, 1024, 512, 128, help="越低越快，越高越准")
        k_num = st.slider("K-Means 聚类数", 2, 8, 5, help="色块分割的颜色数量")
    with st.expander("📐 构图参数"):
        t_diag = st.slider("对角线容差 (Slope)", 0.1, 0.5, 0.3, 0.05)
        t_thirds = st.slider("三分法容差 (Slope)", 0.1, 0.5, 0.2, 0.05)
        t_sym = st.slider("对称性容差 (Threshold)", 50.0, 200.0, 120.0, 10.0, help="RGB欧氏距离容忍上限")
        t_sym_blur = st.slider("对称模糊强度 (Blur K)", 1, 51, 31, 2, help="越高越忽略细节，仅看大色块平衡")
    with st.expander("🎨 色彩参数"):
        t_clarity = st.slider("高光阈值 (Clarity)", 0.5, 0.9, 0.7, 0.05, help="定义'清晰'的最低亮度")
    with st.expander("🌗 图底 & 文字"):
        ref_tex = st.slider("纹理归一化基准", 10.0, 100.0, 50.0, help="Sobel 能量差的分母")
        t_text = st.slider("文字及格线", 40.0, 80.0, 60.0, help="低于此分数的文字会被标红")
    with st.expander("⚖️ 评分权重"):
        w1 = st.number_input("构图权重", 0.0, 1.0, 0.3, 0.1)
        w2 = st.number_input("色彩权重", 0.0, 1.0, 0.3, 0.1)
        w3 = st.number_input("图底权重", 0.0, 1.0, 0.4, 0.1)
    config = {
        'process_width': int(p_width),
        'seg_kmeans_k': int(k_num),
        'comp_diag_slope': float(t_diag),
        'comp_thirds_slope': float(t_thirds),
        'comp_sym_tolerance': float(t_sym),
        'comp_sym_blur_k': int(t_sym_blur),
        'color_clarity_thresh': float(t_clarity),
        'fg_tex_norm': float(ref_tex),
        'text_score_thresh': float(t_text),
        'weight_composition': float(w1),
        'weight_color': float(w2),
        'weight_figure_ground': float(w3)
    }

# === 主界面 ===
st.title("图片参数获取工具")

uploaded_file = st.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image_pil = Image.open(uploaded_file)
    img_np = np.array(image_pil.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    @st.cache_resource
    def get_engine():
        return OmniVisualEngine()

    engine = get_engine()

    with st.spinner("AI 正在扫描全图特征..."):
        data = engine.analyze(img_bgr, config=config)

    report = AestheticDiagnostician.generate_report(data, config=config)

    st.divider()
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
            
    if report['suggestions']:
        with st.expander("🛠️ 点击查看优化建议 (Action Items)", expanded=True):
            for item in report['suggestions']:
                st.warning(item)
    st.divider()

    # === 布局设计 ===
    
    # 顶部：原图 + 雷达图
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
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False, 'staticPlot': False}
        )

    st.divider()

    # 底部：详细数据表格 (Tabs)
    tab1, tab2, tab3 = st.tabs(["📐 构图", "🎨 色彩", "🌗 图底 & 文字"])
    
    # --- Tab 1: 构图 ---
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("对角线主导 (范围: 0-100)", f"{data.composition_diagonal}", help="主体是否靠近对角线")
        c2.metric("三分法契合 (范围: 0-100)", f"{data.composition_thirds}", help="主体是否靠近黄金分割点")
        c3.metric("物理分布平衡 (范围: 0-100)", f"{data.composition_balance}", help="左右物体面积力矩平衡度")
        
        # 兼容性处理
        sym_score = float(getattr(data, 'composition_symmetry', getattr(data, 'color_symmetry', 0)))
        c4.metric("视觉色彩平衡 (范围: 0-100)", f"{sym_score}", help="左右色彩镜像对称度 (RGB欧氏距离)")
        
        st.caption("🔬 构图逻辑可视化诊断")
        img_c1, img_c2, img_c3, img_c4 = st.columns(4)
        with img_c1:
            if getattr(data, 'vis_diag', None) is not None:
                st.image(data.vis_diag, use_container_width=True, caption="对角线辅助线")
            else:
                st.caption("无数据")
        with img_c2:
            if getattr(data, 'vis_thirds', None) is not None:
                st.image(data.vis_thirds, use_container_width=True, caption="三分法连线")
            else:
                st.caption("无数据")
        with img_c3:
            if getattr(data, 'vis_balance', None) is not None:
                st.image(data.vis_balance, use_container_width=True, caption="左右力矩分布")
            else:
                st.caption("无数据")
        with img_c4:
            if getattr(data, 'vis_symmetry_heatmap', None) is not None:
                st.image(data.vis_symmetry_heatmap, use_container_width=True, caption="镜像色差热力图")
            else:
                st.caption("无数据")
        

    # --- Tab 2: 色彩 ---
    with tab2:
        st.markdown("#### 🎨 色彩与光影分析")
        c1, c2 = st.columns([1.5, 1])
        with c1:
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.metric("暖色调占比 (范围: 0-100%)", f"{data.color_warmth:.0%}", help="Luv v>0 区域占比")
                if getattr(data, 'vis_warmth', None) is not None:
                    st.image(data.vis_warmth, use_container_width=True, caption="分布图(红暖蓝冷)")
            with r1c2:
                st.metric("平均饱和度 (范围: 0.0-1.0 Chroma)", f"{data.color_saturation:.2f}", help="Chroma 平均值")
                if getattr(data, 'vis_saturation', None) is not None:
                    st.image(data.vis_saturation, use_container_width=True, caption="热力图(红高蓝低)")
            
            st.divider()
            
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                st.metric("平均亮度 (范围: 0-1 L)", f"{data.color_brightness:.2f}", help="L 通道均值")
                if getattr(data, 'vis_brightness', None) is not None:
                    st.image(data.vis_brightness, use_container_width=True, caption="亮度分布(人眼灰度)")
            with r2c2:
                st.metric("光影对比度 (范围: 0.0-0.5+ StdDev)", f"{data.color_contrast:.2f}", help="L 通道标准差")
                if getattr(data, 'vis_contrast', None) is not None:
                    st.image(data.vis_contrast, use_container_width=True, caption="明暗色阶(黑/灰/白)")
        
        with c2:
            st.metric("高亮区域占比 (范围: 0-100%)", f"{data.color_clarity:.0%}", help="亮度 > 0.7 的像素占比")
            if getattr(data, 'vis_clarity', None) is not None:
                st.image(data.vis_clarity, use_container_width=True, caption="清晰度分布图 (聚光灯效果)")
            score_clarity = data.color_clarity
            if score_clarity > 0.85:
                st.error("💥 **严重过曝**：高光溢出，画面细节丢失，视觉刺眼。")
            elif score_clarity > 0.3:
                st.success("☀️ **通透清晰**：画面有充足的高光区域，视觉传达效率高。")
            elif score_clarity > 0.1:
                st.info("☁️ **柔和/自然**：光照分布均匀，可能具有电影感或胶片感。")
            else:
                st.warning("🌫️ **沉闷/雾感**：高光缺失，画面可能显得灰暗或对焦不清。")

    # --- Tab 3: 图底 & 文字 ---
    with tab3:
        st.markdown("#### 🔤 文字易读性诊断")
        if getattr(data, 'fg_text_present', False):
            c1, c2 = st.columns([1.5, 1])
            with c1:
                if getattr(data, 'vis_text_analysis', None) is not None:
                    st.image(data.vis_text_analysis, use_container_width=True, caption="易读性诊断 (数字为综合评分)")
            with c2:
                st.metric("文字综合易读性 (范围: 0-100)", f"{data.fg_text_legibility}/100", 
                          delta=("优秀" if data.fg_text_legibility > 80 else "需优化"))
                st.metric("平均对比度 (范围: 0-200+ ΔE)", f"{data.fg_text_contrast:.1f}", help="字重色彩与背景的差异度")
                st.info("""
                诊断图例：
                - 🟩 绿框：易读 (Score > 60)，图底关系良好
                - 🟥 红框：难辨 (Score < 60)，对比度低或背景杂乱
                """)
                if data.fg_text_legibility < 60:
                    st.warning("⚠️ 建议：给红框内的文字添加阴影、描边或半透明底板。")
        else:
            st.info("ℹ️ 画面中未检测到明显文字。")

        st.divider()
        
        st.markdown("#### 🖼️ 图形主体分析")
        c1, c2, c3 = st.columns(3)
        c1.metric("面积差异 (范围: 0.0-1.0)", f"{data.fg_area_diff:.2f}")
        c2.metric("色彩差异 (范围: 0-200+ ΔE)", f"{data.fg_color_diff:.1f}")
        c3.metric("纹理差异 (范围: 0.0-1.0)", f"{data.fg_texture_diff:.3f}")
        
        st.markdown("#### 🔬 视觉处理过程可视化")
        vc1, vc2, vc3 = st.columns([1, 1, 1.5])
        with vc1:
            st.caption("1. AI 主体分割 (Mask)")
            _mask = getattr(data, 'vis_mask', None)
            if _mask is not None:
                # 兼容性处理：mask 可能是 bool 或 0-1 或 0-255
                mask_display = (_mask.astype(np.uint8) * 255) if _mask.max() <= 1 else _mask
                st.image(mask_display, use_container_width=True, clamp=True)
            else:
                st.warning("未生成")
        with vc2:
            st.caption("2. 平均色彩抽离 (Color)")
            _color = getattr(data, 'vis_color_contrast', None)
            if _color is not None:
                st.image(_color, use_container_width=True, caption=f"Diff: {data.fg_color_diff}")
                if data.fg_color_diff > 100:
                    st.caption("✅ 强对比 (撞色)")
                elif data.fg_color_diff < 50:
                    st.caption("⚠️ 弱对比 (顺色)")
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
        
        # 调试区
        with st.expander("查看单通道边缘图 (用于调试)"):
            ec1, ec2 = st.columns(2)
            with ec1:
                _fg = getattr(data, 'vis_edge_fg', None)
                if _fg is not None:
                    st.image(_fg, use_container_width=True, caption="前景边缘", clamp=True)
            with ec2:
                _bg = getattr(data, 'vis_edge_bg', None)
                if _bg is not None:
                    st.image(_bg, use_container_width=True, caption="背景边缘", clamp=True)

    # === 批量处理模块 ===
    st.divider()
    st.header("📦 批量分析与导出")
    batch_files = st.file_uploader("批量上传图片", type=["jpg","jpeg","png"], accept_multiple_files=True)
    
    if batch_files:
        max_files = 50
        if len(batch_files) > max_files:
            st.warning(f"已选择 {len(batch_files)} 张，超出上限 {max_files}，将仅处理前 {max_files} 张。")
            batch_files = batch_files[:max_files]
            
        tex_ref = st.slider("纹理归一化参考值", min_value=10.0, max_value=100.0, value=50.0, step=1.0)
        run = st.button("开始批量分析")
        
        if run:
            # 临时调整引擎参数 (如果有的话)
            if hasattr(engine, 'ref_max_texture'):
                engine.ref_max_texture = tex_ref
                
            rows = []
            zip_buffer = io.BytesIO()
            zf = zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED)
            progress = st.progress(0)
            size_limit = 10 * 1024 * 1024 # 10MB
            
            for idx, f in enumerate(batch_files):
                try:
                    # 检查文件大小
                    file_size = getattr(f, 'size', None)
                    if file_size is None:
                        try: file_size = f.getbuffer().nbytes
                        except: file_size = None
                        
                    if file_size is not None and file_size > size_limit:
                        rows.append({"filename": f.name, "error": "文件过大，已跳过", "processed": False})
                        progress.progress(int((idx+1)/len(batch_files)*100))
                        continue
                        
                    # 读取图片
                    img_np_bgr = cv2.cvtColor(np.array(Image.open(f).convert('RGB')), cv2.COLOR_RGB2BGR)
                    
                    # 分析
                    d = engine.analyze(img_np_bgr)
                    rep = AestheticDiagnostician.generate_report(d)
                    
                    # 汇总问题描述
                    def p(d):
                        items = []
                        if d.color_clarity < 0.1: items.append("清晰度低")
                        if d.color_contrast < 0.15: items.append("对比度低")
                        if d.fg_color_diff < 40: items.append("色彩分离度弱")
                        if d.composition_balance < 40: items.append("平衡失衡")
                        if getattr(d,'fg_text_present',False) and d.fg_text_legibility < 60: items.append("文字难辨")
                        return "、".join(items) if items else "无明显问题"
                    
                    # 添加数据行
                    rows.append({
                        "filename": f.name,
                        "score_total": rep["total_score"],
                        "rating": rep["rating_level"],
                        "diag": d.composition_diagonal,
                        "thirds": d.composition_thirds,
                        "balance": d.composition_balance,
                        "symmetry": float(getattr(d,'composition_symmetry', getattr(d,'color_symmetry',0.0))),
                        "warmth": d.color_warmth,
                        "warmth_pct": round(d.color_warmth * 100.0, 1),
                        "saturation": d.color_saturation,
                        "saturation_pct": round(d.color_saturation * 100.0, 1),
                        "brightness": d.color_brightness,
                        "brightness_pct": round(d.color_brightness * 100.0, 1),
                        "contrast": d.color_contrast,
                        "contrast_pct": round(d.color_contrast * 100.0, 1),
                        "clarity": d.color_clarity,
                        "clarity_pct": round(d.color_clarity * 100.0, 1),
                        "fg_area_diff": d.fg_area_diff,
                        "fg_area_pct": round(d.fg_area_diff * 100.0, 1),
                        "fg_color_diff": d.fg_color_diff,
                        "fg_color_diff_norm": round(min(100.0, (float(d.fg_color_diff) / 100.0) * 100.0), 1),
                        "fg_texture_diff": d.fg_texture_diff,
                        "fg_texture_pct": round(d.fg_texture_diff * 100.0, 1),
                        "text_present": getattr(d, 'fg_text_present', False),
                        "text_legibility": getattr(d, 'fg_text_legibility', 0.0),
                        "problems": p(d),
                        "processed": True,
                        "error": None
                    })
                    
                    # 保存诊断图到 ZIP
                    def add_png(name, arr):
                        if arr is None: return
                        img = Image.fromarray(arr)
                        bio = io.BytesIO()
                        img.save(bio, format='PNG')
                        zf.writestr(name, bio.getvalue())
                        
                    base = f.name.rsplit('.',1)[0]
                    add_png(f"{base}_diag.png", getattr(d,'vis_diag', None))
                    add_png(f"{base}_thirds.png", getattr(d,'vis_thirds', None))
                    add_png(f"{base}_balance.png", getattr(d,'vis_balance', None))
                    add_png(f"{base}_symmetry.png", getattr(d,'vis_symmetry_heatmap', None))
                    add_png(f"{base}_clarity.png", getattr(d,'vis_clarity', None))
                    add_png(f"{base}_warmth.png", getattr(d,'vis_warmth', None))
                    add_png(f"{base}_saturation.png", getattr(d,'vis_saturation', None))
                    add_png(f"{base}_brightness.png", getattr(d,'vis_brightness', None))
                    add_png(f"{base}_contrast.png", getattr(d,'vis_contrast', None))
                    add_png(f"{base}_edges.png", getattr(d,'vis_edge_composite', None))
                    add_png(f"{base}_text.png", getattr(d,'vis_text_analysis', None))
                    
                    progress.progress(int((idx+1)/len(batch_files)*100))
                    
                except Exception as e:
                    rows.append({
                        "filename": f.name,
                        "error": str(e),
                        "processed": False
                    })
                    progress.progress(int((idx+1)/len(batch_files)*100))
                    continue
                    
            zf.close()
            
            # 显示结果
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button("下载分析数据 CSV", data=csv_bytes, file_name="analysis.csv", mime="text/csv")
            st.download_button("下载诊断图片 ZIP", data=zip_buffer.getvalue(), file_name="diagnostics.zip", mime="application/zip")