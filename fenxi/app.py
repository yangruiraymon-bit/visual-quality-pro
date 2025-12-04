import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import pandas as pd
import io
import zipfile
import time
import json

# 尝试导入核心模块
try:
    from omni_engine import OmniVisualEngine, AestheticDiagnostician, BenchmarkManager
    # 导入新拆分的服务模块
    from benchmark_service import BenchmarkTrainer
except ImportError as e:
    st.error(f"❌ 缺少核心模块: {e}。请确保 omni_engine.py 和 benchmark_service.py 在同一目录下。")
    st.stop()

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(page_title="全能视觉分析 Pro (服务架构版)", layout="wide", page_icon="🧿")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 5rem; }
        h1 { font-size: 2.0rem !important; margin-bottom: 0.5rem !important; }
        section[data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #ddd; }
        .stButton button { border-radius: 8px; font-weight: 600; }
        .stFileUploader { padding: 1.5rem; border: 2px dashed #e0e0e0; border-radius: 12px; background-color: #ffffff; }
        
        /* --- 指标卡样式 --- */
        [data-testid="stMetric"] {
            background-color: #f8f9fa;
            padding: 10px 15px;
            border-radius: 8px;
            border: 1px solid #eee;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            transition: all 0.2s;
        }
        [data-testid="stMetric"]:hover {
            border-color: #d1d5db;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 状态管理
# ==========================================
if 'batch_df' not in st.session_state: st.session_state.batch_df = None
if 'batch_zip' not in st.session_state: st.session_state.batch_zip = None
if 'batch_imgs' not in st.session_state: st.session_state.batch_imgs = [] 
if 'batch_logs' not in st.session_state: st.session_state.batch_logs = []
if 'processing' not in st.session_state: st.session_state.processing = False
if 'benchmark_profile' not in st.session_state: st.session_state.benchmark_profile = None

@st.cache_resource
def get_engine():
    return OmniVisualEngine()

engine = get_engine()

# ==========================================
# 3. 核心工具函数 (评分逻辑找回)
# ==========================================

def make_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def calculate_dual_score(data, profile, bm_manager):
    """
    核心评分逻辑：支持【单标杆】和【双标杆(正向+负向)】
    """
    # 1. 判断是 新版双标杆结构 还是 旧版单标杆结构
    is_dual = 'positive' in profile and isinstance(profile['positive'], dict)
    
    if is_dual:
        # --- 双向评分逻辑 ---
        # A. 计算正向得分 (Reward)
        res_pos = bm_manager.score_against_benchmark(data, profile['positive'])
        score_pos = res_pos['total_score']
        
        # B. 计算负向得分 (Penalty) - 仅当存在负向配置时
        score_neg = 0
        if 'negative' in profile and profile['negative']:
            res_neg = bm_manager.score_against_benchmark(data, profile['negative'])
            score_neg = res_neg['total_score'] # 这里的"分"代表"有多像烂图"
        
        # C. 综合计算
        # 公式：最终分 = 正向分 - (负向分 * 惩罚系数)
        penalty_factor = 0.4 
        final_score = score_pos - (score_neg * penalty_factor)
        final_score = max(0, min(100, final_score)) # 截断在 0-100
        
        # 评级逻辑
        if final_score >= 90: rating = "S (卓越)"
        elif final_score >= 80: rating = "A (优秀)"
        elif final_score >= 70: rating = "B (良好)"
        elif final_score >= 60: rating = "C (合格)"
        else: rating = "D (不合格)"
        
        return {
            'total_score': final_score,
            'rating_level': rating,
            'mode': '双向标杆',
            'details': res_pos['details'], # 详细维度对比依然用正向的作为基准
            'score_breakdown': {'pos': score_pos, 'neg': score_neg} # 记录细分
        }
    else:
        # --- 传统单标杆逻辑 (兼容旧版) ---
        res = bm_manager.score_against_benchmark(data, profile)
        res['mode'] = '单向标杆'
        res['score_breakdown'] = None
        return res

def normalize_values(source, is_profile=False):
    """雷达图数据归一化"""
    def get(k): 
        val = source.get(k, {}).get('target', 0) if is_profile else getattr(source, k, 0)
        return float(val) if val is not None else 0.0
    
    return [
        get('composition_diagonal'), get('composition_thirds'), get('composition_balance'), get('composition_symmetry'),
        get('color_warmth')*100, get('color_saturation')*100, get('color_brightness')*100, min(100, (get('color_contrast')/0.3)*100), get('color_clarity')*100,
        get('fg_area_diff')*100, min(100, get('fg_color_diff')), get('fg_texture_diff')*100,
        get('fg_text_legibility') if is_profile or getattr(source, 'fg_text_present', False) else 0
    ]

# ==========================================
# 4. 批量处理逻辑
# ==========================================
def to_excel_with_all_images(df, img_dicts):
    """生成包含 13 个维度诊断图的 Excel 文件"""
    output = io.BytesIO()
    img_columns_map = [
        ('v_diag', '图:对角线'),
        ('v_thirds', '图:三分法'),
        ('v_bal', '图:平衡度'),
        ('v_sym', '图:对称性'),
        ('v_sat', '图:饱和度'),
        ('v_bri', '图:亮度'),
        ('v_warm', '图:暖色调'),
        ('v_cont', '图:对比度'),
        ('v_clar', '图:清晰度'),
        ('v_f_col', '图:主体色差'),
        ('v_f_area', '图:主体Mask'),
        ('v_f_tex', '图:纹理对比'),
        ('v_text', '图:文字分析')
    ]
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='分析结果', index=False)
        workbook = writer.book
        worksheet = writer.sheets['分析结果']
        header_fmt = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#DDEBF7', 'valign': 'vcenter', 'align': 'center'})
        base_col = len(df.columns)
        worksheet.set_default_row(80)
        for i, (_, title) in enumerate(img_columns_map):
            worksheet.write(0, base_col + i, title, header_fmt)
            worksheet.set_column(base_col + i, base_col + i, 18)
        for row_idx, img_dict in enumerate(img_dicts):
            if not img_dict:
                continue
            excel_row = row_idx + 1
            for i, (key, _) in enumerate(img_columns_map):
                img_raw_bytes = img_dict.get(key)
                if img_raw_bytes:
                    image_stream = io.BytesIO(img_raw_bytes)
                    unique_filename = f"r{row_idx}_{key}.png"
                    try:
                        worksheet.insert_image(
                            excel_row, base_col + i,
                            unique_filename,
                            {
                                'image_data': image_stream,
                                'x_scale': 0.12, 'y_scale': 0.12,
                                'object_position': 1
                            }
                        )
                    except:
                        pass
    return output.getvalue()

def run_batch_process(files, cfg, need_zip, profile=None):
    st.session_state.processing = True
    st.session_state.batch_logs = []
    ALL_DIMS_MAPPING = [
        ('composition_diagonal', '构图_对角线'), ('composition_thirds', '构图_三分法'),
        ('composition_balance', '构图_平衡度'), ('composition_symmetry', '构图_对称性'),
        ('color_saturation', '色彩_饱和度'), ('color_brightness', '色彩_亮度'),
        ('color_warmth', '色彩_暖色调'), ('color_contrast', '色彩_对比度'),
        ('color_clarity', '色彩_清晰度'),
        ('fg_color_diff', '图底_色差'), ('fg_area_diff', '图底_占比'),
        ('fg_texture_diff', '图底_纹理差'), ('fg_text_legibility', '文字_易读性')
    ]
    rows = []
    diff_rows = []
    raw_json_list = []
    img_dicts = []
    zip_buffer = io.BytesIO() if need_zip else None
    zf = zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) if need_zip else None
    bm_manager = BenchmarkManager() if profile else None
    total = len(files)
    progress_bar = st.progress(0)
    for idx, f in enumerate(files):
        try:
            f.seek(0)
            f_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                rows.append({"文件名": f.name, "综合得分": 0, "评级": "读取失败"})
                diff_rows.append({"文件名": f.name})
                img_dicts.append({})
                continue
            d = engine.analyze(img_bgr, config=cfg)
            if profile:
                if 'positive' in profile:
                    try:
                        from app import calculate_dual_score
                        res = calculate_dual_score(d, profile, bm_manager)
                    except:
                        res = bm_manager.score_against_benchmark(d, profile['positive'])
                        res['mode'] = '标杆(降级)'
                    target_dict = profile['positive']
                else:
                    res = bm_manager.score_against_benchmark(d, profile)
                    target_dict = profile
                final_score = res['total_score']
                final_rating = res['rating_level']
                mode_str = f"标杆 ({res.get('mode','默认')})"
            else:
                rep = AestheticDiagnostician.generate_report(d, config=cfg)
                final_score = rep['total_score']
                final_rating = rep['rating_level']
                mode_str = "通用模式"
                target_dict = {}
            base_info = {"文件名": f.name, "综合得分": final_score, "评级": final_rating, "模式": mode_str}
            row_data = base_info.copy()
            diff_data = base_info.copy()
            for key, label in ALL_DIMS_MAPPING:
                val = getattr(d, key, 0) or 0
                if key == 'fg_text_legibility' and not getattr(d, 'fg_text_present', False): val = 0
                row_data[label] = round(val, 2)
                if profile and key in target_dict:
                    t_val = target_dict[key].get('target', 0)
                    diff_data[f"Δ_{label}"] = round(val - t_val, 2)
                else:
                    diff_data[f"Δ_{label}"] = 0
            rows.append(row_data)
            diff_rows.append(diff_data)
            raw_obj = {k: getattr(d, k) for k, _ in ALL_DIMS_MAPPING}
            raw_obj['filename'] = f.name
            raw_json_list.append(raw_obj)
            curr_imgs = {}
            vis_map = {
                'v_diag': 'vis_diag',
                'v_thirds': 'vis_thirds',
                'v_bal': 'vis_balance',
                'v_sym': 'vis_symmetry_heatmap',
                'v_sat': 'vis_saturation',
                'v_bri': 'vis_brightness',
                'v_warm': 'vis_warmth',
                'v_cont': 'vis_contrast',
                'v_clar': 'vis_clarity',
                'v_f_col': 'vis_color_contrast',
                'v_f_area': 'vis_mask',
                'v_f_tex': 'vis_edge_composite',
                'v_text': 'vis_text_analysis'
            }
            for excel_key, attr_name in vis_map.items():
                img_data = getattr(d, attr_name, None)
                if img_data is not None:
                    b = io.BytesIO()
                    if hasattr(img_data, 'dtype') and img_data.dtype != np.uint8:
                        img_data = img_data.astype(np.uint8)
                    Image.fromarray(img_data).save(b, 'PNG')
                    curr_imgs[excel_key] = b.getvalue()
            img_dicts.append(curr_imgs)
            if zf:
                base_name = f.name.rsplit('.', 1)[0]
                for excel_key, img_bytes in curr_imgs.items():
                    zf.writestr(f"diagnostics/{base_name}_{excel_key}.png", img_bytes)
        except Exception as e:
            st.session_state.batch_logs.append(f"Error {f.name}: {e}")
            img_dicts.append({})
        progress_bar.progress((idx + 1) / total)
    if zf: zf.close()
    st.session_state.batch_df = pd.DataFrame(rows)
    st.session_state.batch_diff_df = pd.DataFrame(diff_rows)
    st.session_state.batch_raw_json = raw_json_list
    st.session_state.batch_imgs = img_dicts
    st.session_state.batch_zip = zip_buffer.getvalue() if need_zip else None
    st.session_state.processing = False

# ==========================================
# 5. 侧边栏布局 (修复重复ID版)
# ==========================================
with st.sidebar:
    st.header("🧿 视觉分析 Pro")
    mode = st.radio(
        "工作模式",
        ["📸 单图诊断", "📦 批量工厂", "🏆 建立标杆"],
        index=0,
        key="nav_mode_selection"
    )
    st.divider()
    
    current_profile = st.session_state.benchmark_profile
    if current_profile:
        if 'positive' in current_profile:
            st.success("✅ 双向标杆：已激活")
        else:
            st.success("✅ 单向标杆：已激活")
        if st.button("清除标杆", use_container_width=True):
            st.session_state.benchmark_profile = None; st.rerun()
    
    with st.expander("⚙️ 基础算法参数", expanded=False):
        p_width = st.slider("处理分辨率", 256, 1024, 512, 128, help="越高性能消耗越大")
        k_num = st.slider("色彩聚类数", 2, 8, 5)
        st.caption("阈值微调")
        t_diag = st.slider("对角线判定", 0.1, 0.5, 0.3)
        t_sym_blur = st.slider("对称模糊K", 1, 51, 31, 2)
        ref_tex = st.slider("纹理基准", 10.0, 100.0, 50.0)
        t_clarity = st.slider("高光/清晰阈值", 0.5, 0.9, 0.7)
    
    with st.expander("⚖️ 评分权重与容差", expanded=False):
        st.info("自定义 13 个维度的评分影响因子")
        dims_geo = [('composition_diagonal', '对角线'), ('composition_thirds', '三分法'), ('composition_balance', '平衡度'), ('composition_symmetry', '对称性')]
        dims_color = [('color_saturation', '饱和度'), ('color_brightness', '亮度'), ('color_warmth', '暖色调'), ('color_contrast', '对比度'), ('color_clarity', '清晰度')]
        dims_content = [('fg_color_diff', '主体色差'), ('fg_area_diff', '主体占比'), ('fg_texture_diff', '纹理差异'), ('fg_text_legibility', '文字易读')]
        loaded_weights = {}
        loaded_tols = {}
        if current_profile:
            loaded_weights = current_profile.get('weights', {})
            loaded_tols = current_profile.get('tolerances', {})
        tab_w, tab_t = st.tabs(["📊 权重", "🎯 容差"])
        final_weights = {}
        final_tols = {}
        def render_sliders(tab, category_name, dims, is_weight=True):
            tab.caption(f"**{category_name}**")
            for k, label in dims:
                if is_weight:
                    default_val = float(loaded_weights.get(k, 1.0))
                    key = f"w_{k}"
                    if key in st.session_state and st.session_state[key] > 5.0:
                        st.session_state[key] = default_val
                    val = tab.slider(label, 0.0, 5.0, default_val, 0.1, key=key)
                    final_weights[k] = val
                else:
                    val_from_file = loaded_tols.get(k)
                    if not val_from_file and current_profile and 'positive' in current_profile:
                        if k in current_profile['positive'] and isinstance(current_profile['positive'][k], dict):
                             val_from_file = current_profile['positive'][k].get('tolerance')
                    elif not val_from_file and current_profile and k in current_profile and isinstance(current_profile[k], dict):
                        val_from_file = current_profile[k].get('tolerance')
                    default_val = float(val_from_file) if val_from_file else 0.2
                    max_val = max(1.0, default_val * 2.5)
                    key = f"t_{k}"
                    if key in st.session_state and st.session_state[key] > max_val:
                        st.session_state[key] = default_val
                    val = tab.slider(label, 0.0, max_val, default_val, max_val/50, key=key)
                    final_tols[k] = val
        with tab_w:
            render_sliders(tab_w, "📐 构图", dims_geo, True)
            st.markdown("---")
            render_sliders(tab_w, "🎨 色彩", dims_color, True)
            st.markdown("---")
            render_sliders(tab_w, "🌗 图底", dims_content, True)
        with tab_t:
            render_sliders(tab_t, "📐 构图", dims_geo, False)
            st.markdown("---")
            render_sliders(tab_t, "🎨 色彩", dims_color, False)
            st.markdown("---")
            render_sliders(tab_t, "🌗 图底", dims_content, False)
    config = {
        'process_width': p_width,
        'seg_kmeans_k': k_num,
        'comp_diag_slope': t_diag,
        'comp_sym_blur_k': t_sym_blur,
        'fg_tex_norm': ref_tex,
        'color_clarity_thresh': t_clarity,
        'comp_thirds_slope': 0.2,
        'comp_sym_tolerance': 120.0,
        'text_score_thresh': 60.0,
        'weights': final_weights,
        'tolerances': final_tols
    }

# ==========================================
# 6. 主界面逻辑
# ==========================================

# --- 模式 1: 批量工厂 (UI 更新) --- 
if mode == "📦 批量工厂": 
    st.title("📦 批量处理中心") 
    if st.session_state.benchmark_profile: 
        st.subheader("当前标准：🏆 行业标杆匹配度检测") 
    else: 
        st.subheader("当前标准：🌐 通用美学质量评分") 
    
    with st.container(): 
        batch_files = st.file_uploader("📂 选择图片", type=["jpg","png","jpeg"], accept_multiple_files=True) 
    
    if batch_files: 
        st.divider() 
        c1, c2, c3 = st.columns([2, 1, 1]) 
        with c1: st.info(f"已加载 **{len(batch_files)}** 张图片") 
        with c2: opt_zip = st.checkbox("生成全套图包", value=True, help="包含所有中间过程的诊断图 (构图、热力、Mask等)") 
        with c3: 
            st.button("🚀 开始批量分析", type="primary", use_container_width=True, 
                      on_click=run_batch_process, 
                      args=(batch_files, config, opt_zip, st.session_state.benchmark_profile)) 
    
    if st.session_state.processing: 
        st.divider(); st.warning("⏳ 正在进行全维度分析，请稍候...") 
        with st.expander("实时日志"): st.text("\n".join(st.session_state.batch_logs[-10:])) 
    
    if st.session_state.batch_df is not None: 
        st.divider() 
        st.subheader("3. 结果交付 (全维度)") 
        st.success(f"✅ 处理完成！已生成 13 维度完整数据。") 
        tab_main, tab_diff = st.tabs(["📋 完整得分表", "📊 标杆偏差表 (Diff)"]) 
        with tab_main: 
            st.dataframe(st.session_state.batch_df, use_container_width=True, height=400) 
        with tab_diff: 
            if 'batch_diff_df' in st.session_state: 
                st.dataframe(st.session_state.batch_diff_df.style.background_gradient(cmap="RdBu_r", vmin=-50, vmax=50), use_container_width=True, height=400) 
            else: 
                st.info("需要加载标杆模型才能查看偏差表。") 
        st.divider() 
        st.markdown("### 📥 下载中心") 
        d1, d2, d3, d4 = st.columns(4) 
        with d1: 
            if st.session_state.batch_imgs: 
                excel_data = to_excel_with_all_images(st.session_state.batch_df, st.session_state.batch_imgs) 
                st.download_button("📊 完整报表 (Excel+图)", excel_data, "Report_Full_Visual.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True) 
        with d2: 
            if 'batch_diff_df' in st.session_state: 
                st.download_button("📉 偏差分析表 (Diff CSV)", st.session_state.batch_diff_df.to_csv(index=False).encode('utf-8-sig'), "Report_Diff_Analysis.csv", "text/csv", use_container_width=True) 
        with d3: 
            if 'batch_raw_json' in st.session_state: 
                json_str = json.dumps(st.session_state.batch_raw_json, default=make_serializable, indent=4) 
                st.download_button("⚙️ 原始参数 (JSON)", json_str, "Raw_Parameters.json", "application/json", use_container_width=True) 
        with d4: 
            if st.session_state.batch_zip: 
                st.download_button("📦 诊断图包 (ZIP)", st.session_state.batch_zip, "Diagnostic_Images.zip", "application/zip", use_container_width=True) 
    elif not batch_files: 
        st.divider(); st.caption("👈 请先上传图片开始工作...")

# --- 模式 2: 单图诊断 (完整修复版：13指标 + 全套诊断图) ---
elif mode == "📸 单图诊断":
    st.title("🧿 单图深度诊断")
    uploaded_file = st.file_uploader("上传图片", type=['jpg','png','jpeg'])
    
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

        # 1. AI 分析与评分
        with st.spinner("AI 正在进行全维度扫描..."):
            data = engine.analyze(img_bgr, config=config)
            
            is_bench = st.session_state.benchmark_profile is not None
            bench_details = {}
            
            if is_bench:
                bm = BenchmarkManager()
                # 双向评分
                res = calculate_dual_score(data, st.session_state.benchmark_profile, bm)
                final_score = res['total_score']
                final_rating = res['rating_level']
                bench_details = res['details']
                score_breakdown = res.get('score_breakdown')
                mode_display = res.get('mode', '标杆')
            else:
                rep = AestheticDiagnostician.generate_report(data, config=config)
                final_score, final_rating = rep['total_score'], rep['rating_level']
                mode_display = "通用"
                score_breakdown = None

        # 2. 界面布局
        c1, c2 = st.columns([1, 1.2])
        
        # --- 左列：核心数据与指标卡 ---
        with c1:
            st.image(image_pil, use_container_width=True)
            
            st.metric("🏆 综合得分", f"{final_score:.1f}", delta=f"{final_rating} ({mode_display})")
            
            if score_breakdown:
                pos = score_breakdown['pos']
                neg = score_breakdown['neg']
                st.info(f"✅ 正向拟合: {pos:.1f} | ⛔ 负向排斥: {neg:.1f}")

            st.divider()

            def smart_card(col, label, key, unit="", multiplier=1.0):
                raw_val = getattr(data, key, 0)
                if raw_val is None: raw_val = 0
                
                if is_bench and key in bench_details:
                    item = bench_details[key]
                    score = item['score']
                    target = item['target'] * multiplier
                    actual = item['actual'] * multiplier
                    
                    if score >= 80: state = "normal"
                    elif score >= 60: state = "off"
                    else: state = "inverse"
                    
                    col.metric(
                        label,
                        f"{score:.0f}分",
                        f"实测{actual:.1f}{unit} / 标杆{target:.1f}",
                        delta_color=state
                    )
                else:
                    col.metric(label, f"{raw_val*multiplier:.1f}{unit}")

            st.caption("📐 构图几何")
            r1a, r1b = st.columns(2)
            smart_card(r1a, "对角线", "composition_diagonal")
            smart_card(r1b, "三分法", "composition_thirds")
            r1c, r1d = st.columns(2)
            smart_card(r1c, "平衡度", "composition_balance")
            smart_card(r1d, "对称性", "composition_symmetry")

            st.caption("🎨 色彩氛围")
            r2a, r2b, r2c = st.columns(3)
            smart_card(r2a, "饱和度", "color_saturation", "%", 100)
            smart_card(r2b, "亮度", "color_brightness", "%", 100)
            smart_card(r2c, "暖色调", "color_warmth", "%", 100)
            r2d, r2e = st.columns(2)
            smart_card(r2d, "对比度", "color_contrast", "", 1.0)
            smart_card(r2e, "清晰度", "color_clarity", "%", 100)

            st.caption("🌗 图底与信息")
            r3a, r3b = st.columns(2)
            smart_card(r3a, "主体色差", "fg_color_diff")
            smart_card(r3b, "主体占比", "fg_area_diff", "%", 100)
            r3c, r3d = st.columns(2)
            smart_card(r3c, "纹理差异", "fg_texture_diff")
            if getattr(data, 'fg_text_present', False):
                smart_card(r3d, "文字易读", "fg_text_legibility")
            else:
                r3d.metric("文字", "无", delta_color="off")

        # --- 右列：可视化诊断图表 ---
        with c2:
            st.subheader("📊 维度雷达")
            cats = ['对角线','三分法','平衡','对称','饱和','亮度','暖色','对比','清晰','色差','占比','纹理','易读']
            vals = normalize_values(data, False)
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself', name='当前图片', line_color='#3498db'))
            
            if is_bench:
                bench_vals = normalize_values(st.session_state.benchmark_profile['positive'] if 'positive' in st.session_state.benchmark_profile else st.session_state.benchmark_profile, True)
                fig.add_trace(go.Scatterpolar(r=bench_vals, theta=cats, fill='toself', name='标杆基准', line_color='#2ecc71', opacity=0.4))
                
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=350, margin=dict(t=20, b=20, l=40, r=40))
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("🔎 AI 视觉诊断图谱")
            
            t_comp, t_color, t_content, t_curve = st.tabs(["📐 构图视角", "🎨 色彩热力", "🌗 主体识别", "🔬 Luv 分布曲线"]) 
            
            with t_comp:
                c_t1, c_t2 = st.columns(2)
                if getattr(data, 'vis_diag', None) is not None:
                    c_t1.image(data.vis_diag, caption="对角线引导", use_container_width=True)
                if getattr(data, 'vis_thirds', None) is not None:
                    c_t2.image(data.vis_thirds, caption="三分法参考", use_container_width=True)
                
                c_t3, c_t4 = st.columns(2)
                if getattr(data, 'vis_balance', None) is not None:
                    c_t3.image(data.vis_balance, caption="视觉平衡点", use_container_width=True)
                if getattr(data, 'vis_symmetry_heatmap', None) is not None:
                    c_t4.image(data.vis_symmetry_heatmap, caption="对称性热力图", use_container_width=True)

            with t_color:
                c_c1, c_c2 = st.columns(2)
                if getattr(data, 'vis_warmth', None) is not None:
                    c_c1.image(data.vis_warmth, caption="暖色调分布", use_container_width=True)
                if getattr(data, 'vis_saturation', None) is not None:
                    c_c2.image(data.vis_saturation, caption="饱和度分布", use_container_width=True)
                
                c_c3, c_c4 = st.columns(2)
                if getattr(data, 'vis_brightness', None) is not None:
                    c_c3.image(data.vis_brightness, caption="亮度分布", use_container_width=True)
                if getattr(data, 'vis_clarity', None) is not None:
                    c_c4.image(data.vis_clarity, caption="清晰度/边缘", use_container_width=True)

            with t_content:
                c_f1, c_f2 = st.columns(2)
                if getattr(data, 'vis_mask', None) is not None:
                    c_f1.image(data.vis_mask, caption="主体分割 Mask", use_container_width=True)
                if getattr(data, 'vis_color_contrast', None) is not None:
                    c_f2.image(data.vis_color_contrast, caption="色彩抽离对比", use_container_width=True)
                
                c_f3, c_f4 = st.columns(2)
                if getattr(data, 'vis_edge_composite', None) is not None:
                    c_f3.image(data.vis_edge_composite, caption="纹理复杂度对比", use_container_width=True)
                
                if getattr(data, 'fg_text_present', False) and getattr(data, 'vis_text_analysis', None) is not None:
                    c_f4.image(data.vis_text_analysis, caption="文字区域检测", use_container_width=True)
                elif not getattr(data, 'fg_text_present', False):
                    c_f4.info("未检测到显著文字")

            with t_curve:
                st.markdown("#### 🔬 Luv 空间感知分布")
                st.caption("基于 CIE Luv 感知均匀色彩空间，使用 推土机距离 (EMD) 对比分布形态。")
                from luv_analysis import LUVAnalysisEngine
                from histogram_scorer import DistributionScorer
                luv_engine = LUVAnalysisEngine()
                scorer = DistributionScorer()
                curr_luv = luv_engine.extract_luv_distributions(img_bgr)
                bench_luv = st.session_state.benchmark_profile.get('luv_curves') if is_bench else None
                scores = None
                if is_bench and bench_luv:
                    scores = scorer.evaluate_luv_quality(curr_luv, bench_luv)
                def plot_luv_curve(title, y_curr, y_bench=None, color="#333"):
                    fig = go.Figure()
                    x_axis = np.linspace(0, 100, len(y_curr))
                    fig.add_trace(go.Scatter(x=x_axis, y=y_curr, mode='lines', fill='tozeroy', name='当前', line=dict(color=color, width=2)))
                    if y_bench is not None:
                        fig.add_trace(go.Scatter(x=x_axis, y=y_bench, mode='lines', name='标杆', line=dict(color='gray', width=2, dash='dash')))
                    fig.update_layout(title=dict(text=title, font=dict(size=14)), xaxis=dict(showgrid=False, title="强度 %"), yaxis=dict(showgrid=False, showticklabels=False), height=200, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
                    return fig
                c_l, c_c, c_h = st.columns(3)
                with c_l:
                    title = "**💡 感知亮度 (L)**"
                    if scores:
                        s = scores['score_L']
                        color = "green" if s > 80 else "red"
                        title += f" <span style='color:{color}; font-size:1.2em'>{s:.0f}分</span>"
                    st.markdown(title, unsafe_allow_html=True)
                    st.plotly_chart(plot_luv_curve("", curr_luv['dist_L'], bench_luv['dist_L'] if bench_luv else None, "#f1c40f"), use_container_width=True)
                    st.caption("波峰靠右=高调；波峰靠左=低调；双峰=高对比。")
                with c_c:
                    title = "**🎨 感知彩度 (C)**"
                    if scores:
                        s = scores['score_C']
                        color = "green" if s > 80 else "red"
                        title += f" <span style='color:{color}; font-size:1.2em'>{s:.0f}分</span>"
                    st.markdown(title, unsafe_allow_html=True)
                    st.plotly_chart(plot_luv_curve("", curr_luv['dist_C'], bench_luv['dist_C'] if bench_luv else None, "#e74c3c"), use_container_width=True)
                    st.caption("衡量色彩的纯度/鲜艳度。")
                with c_h:
                    title = "**🌈 感知色相 (H)**"
                    if scores:
                        s = scores['score_H']
                        color = "green" if s > 80 else "red"
                        title += f" <span style='color:{color}; font-size:1.2em'>{s:.0f}分</span>"
                    st.markdown(title, unsafe_allow_html=True)
                    st.plotly_chart(plot_luv_curve("", curr_luv['dist_H'], bench_luv['dist_H'] if bench_luv else None, "#9b59b6"), use_container_width=True)
                    st.caption("色彩在光谱中的位置分布。")
                if scores:
                    st.info(f"✨ **色彩质感综合得分：{scores['avg_score']:.1f}** (基于 Luv 分布形态相似度计算)")

# --- 模式 3: 建立标杆 (修复加载报错版) --- 
elif mode == "🏆 建立标杆": 
    st.title("🏆 建立行业视觉标杆 (Service版)") 
    
    # --- [核心修复] 定义加载配置的回调函数 --- 
    def on_load_config(): 
        uploaded = st.session_state.get('conf_uploader') 
        if uploaded is not None: 
            try: 
                uploaded.seek(0) 
                p = json.load(uploaded) 
                st.session_state.benchmark_profile = p 
                if 'weights' in p: 
                    for k, v in p['weights'].items(): 
                        st.session_state[f"w_{k}"] = float(v) 
                if 'tolerances' in p: 
                    for k, v in p['tolerances'].items(): 
                        st.session_state[f"t_{k}"] = float(v) 
                st.session_state['_load_msg'] = f"✅ 配置已成功加载: {uploaded.name}" 
            except Exception as e: 
                st.session_state['_load_msg'] = f"❌ 文件解析错误: {str(e)}" 

    st.file_uploader( 
        "📂 加载配置文件", 
        type=["json"], 
        key="conf_uploader", 
        on_change=on_load_config 
    ) 
    
    if '_load_msg' in st.session_state: 
        if "❌" in st.session_state['_load_msg']: 
            st.error(st.session_state['_load_msg']) 
        else: 
            st.success(st.session_state['_load_msg']) 
        del st.session_state['_load_msg'] 
    
    st.divider() 
    
    c_high, c_low = st.columns(2) 
    with c_high: 
        st.subheader("👍 正向标杆 (High)") 
        files_high = st.file_uploader("选择 High 图片", accept_multiple_files=True, key="up_high") 
    with c_low: 
        st.subheader("👎 负向标杆 (Low)") 
        files_low = st.file_uploader("选择 Low 图片", accept_multiple_files=True, key="up_low") 

    use_auto_weight = st.checkbox("🤖 启用自动权重推算 (推荐)", value=True) 

    def call_training_service(f_pos, f_neg, cfg, auto_w): 
        trainer = BenchmarkTrainer() 
        try: 
            status_box = st.empty() 
            status_box.info("🚀 正叫调用训练服务...") 
            profile, dist_data, stats = trainer.train(pos_files=f_pos, neg_files=f_neg, config=cfg, auto_weight_enable=auto_w) 
            st.session_state.benchmark_profile = profile 
            st.session_state['benchmark_dist_data'] = dist_data 
            if auto_w and 'weights' in profile: 
                for k, v in profile['weights'].items(): st.session_state[f"w_{k}"] = float(v) 
            if 'tolerances' in profile: 
                for k, v in profile['tolerances'].items(): st.session_state[f"t_{k}"] = float(v) 
            status_box.empty() 
            st.session_state['_train_msg'] = f"✅ 训练成功! 正向:{stats['pos_count']}, 负向:{stats['neg_count']}" 
        except Exception as e: st.error(f"训练服务出错: {str(e)}") 

    if files_high: 
        st.button("🚀 调用服务开始训练", type="primary", use_container_width=True, 
                  on_click=call_training_service, args=(files_high, files_low, config, use_auto_weight)) 
    
    if '_train_msg' in st.session_state: 
        st.success(st.session_state['_train_msg']); del st.session_state['_train_msg'] 
        
    if st.session_state.benchmark_profile: 
        st.divider(); st.subheader("📊 训练分析") 
        if 'benchmark_dist_data' in st.session_state: 
            with st.expander("📈 查看特征分布 (箱线图)", expanded=True): 
                dist_data = st.session_state['benchmark_dist_data'] 
                fig = go.Figure() 
                for k, vals in dist_data.items(): 
                    w_val = config['weights'].get(k, 1.0) 
                    color = '#2ecc71' if w_val >= 2.5 else '#3498db' 
                    fig.add_trace(go.Box(y=vals, name=k, marker_color=color, boxpoints='all', jitter=0.3)) 
                fig.update_layout(height=400, showlegend=False, margin=dict(t=20,b=20)) 
                st.plotly_chart(fig, use_container_width=True) 

        final_pkg = st.session_state.benchmark_profile.copy() 
        final_pkg['weights'] = config['weights'] 
        final_pkg['tolerances'] = config['tolerances'] 
        json_str = json.dumps(final_pkg, default=make_serializable, indent=4) 
        st.download_button("📦 下载完整配置", json_str, "benchmark_service_output.json", "application/json", type="primary")