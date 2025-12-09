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
import gc
import os
import base64
from pathlib import Path

# 尝试导入核心模块
try:
    from omni_engine import OmniVisualEngine, AestheticDiagnostician, BenchmarkManager, DEFAULT_ANALYSIS_PROMPT
    from benchmark_service import BenchmarkTrainer, AestheticManifold
except ImportError as e:
    st.error(f"❌ 缺少核心模块: {e}。请确保所有 .py 文件在同一目录下。")
    st.stop()

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(page_title="全能视觉分析 Pro (V25.0 VLM Recognition)", layout="wide", page_icon="🧿")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 5rem; }
        h1 { font-size: 2.0rem !important; margin-bottom: 0.5rem !important; }
        .stButton button { border-radius: 8px; font-weight: 600; }
        .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #eee; }
        .kobayashi-tag {
            display: inline-block;
            padding: 4px 12px;
            margin: 2px;
            border-radius: 16px;
            font-size: 0.85em;
            font-weight: 600;
            color: #333;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border: 1px solid #d1d5db;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 状态管理
# ==========================================
if 'batch_df' not in st.session_state: st.session_state.batch_df = None
if 'batch_zip' not in st.session_state: st.session_state.batch_zip = None
if 'batch_imgs_preview' not in st.session_state: st.session_state.batch_imgs_preview = [] 
if 'processing' not in st.session_state: st.session_state.processing = False
if 'benchmark_profile' not in st.session_state: st.session_state.benchmark_profile = None
if 'aesthetic_manifold' not in st.session_state: st.session_state.aesthetic_manifold = None
if 'analysis_prompt' not in st.session_state: st.session_state.analysis_prompt = DEFAULT_ANALYSIS_PROMPT

# ==========================================
# 3. 侧边栏与引擎初始化
# ==========================================
with st.sidebar:
    st.header("🧿 视觉分析 Pro")
    st.caption("内核: VLM (主体/文字识别) + Oklab + CIECAM02")
    
    # VLM 配置
    with st.expander("🧠 视觉大模型 (VLM) 配置", expanded=True):
        _cfg_path = os.path.expanduser("~/.fenxi_vlm.json")
        def _load_vlm():
            try:
                with open(_cfg_path, 'r') as f:
                    c = json.load(f)
                return c.get('api_key', ''), c.get('endpoint', '')
            except Exception:
                return "", ""
        def _save_vlm(k, ep):
            try:
                with open(_cfg_path, 'w') as f:
                    json.dump({"api_key": k, "endpoint": ep}, f, indent=2)
                return True
            except Exception as e:
                st.error(f"保存失败: {e}")
                return False
        def _clear_vlm():
            try:
                if os.path.exists(_cfg_path): os.remove(_cfg_path)
                return True
            except Exception as e:
                st.error(f"清除失败: {e}")
                return False
        _loaded_key, _loaded_ep = _load_vlm()
        vlm_key = st.text_input("Doubao API Key", value=_loaded_key or "", type="password", help="火山引擎 API Key")
        vlm_endpoint = st.text_input("Endpoint ID", value=_loaded_ep or "ep-20250203...", help="方舟平台接入点 ID")
        c_s1, c_s2 = st.columns(2)
        if c_s1.button("保存接入配置", use_container_width=True):
            if _save_vlm(vlm_key, vlm_endpoint):
                st.success("已保存接入配置")
                st.rerun()
        if c_s2.button("清除接入配置", use_container_width=True):
            if _clear_vlm():
                st.success("已清除接入配置")
                st.rerun()
        
        if vlm_key:
            st.success("✅ VLM 已就绪 (用于核心识别)")
        else:
            st.error("⛔ 未配置 VLM: 主体与文字识别将不可用")

    # [Updated] 提示词工程区域 (适配新版识别逻辑)
    with st.expander("📝 提示词配置 (Prompt Config)", expanded=True):
        st.markdown("**VLM 识别指令 (System Prompt)**")
        st.caption("定义 VLM 如何识别画面主体与文字。需严格保持 JSON 输出格式。")
        
        ana_prompt_input = st.text_area(
            "Prompt 内容", 
            value=st.session_state.analysis_prompt, 
            height=300,
            key="ana_prompt_area",
            help="修改此指令可调整 VLM 对主体和文字的识别偏好。"
        )
        
        c_p1, c_p2 = st.columns(2)
        if c_p1.button("💾 保存配置", type="primary", use_container_width=True):
            st.session_state.analysis_prompt = ana_prompt_input
            st.success("已更新！")
            
        if c_p2.button("🔄 重置默认", use_container_width=True):
            st.session_state.analysis_prompt = DEFAULT_ANALYSIS_PROMPT
            st.rerun()

    # 模式选择
    mode = st.radio("工作模式", ["📸 单图诊断", "📦 批量工厂", "🏆 建立标杆"], index=0)
    st.divider()
    
    # 强制刷新
    if st.button("🧹 强制刷新核心引擎"):
        st.cache_resource.clear()
        gc.collect()
        st.rerun()
    
    # 流形模型状态
    if st.session_state.aesthetic_manifold:
        st.success(f"✅ 流形模型：已激活 ({len(st.session_state.aesthetic_manifold.vectors)} 样本)")
        if st.button("清除模型"):
            st.session_state.aesthetic_manifold = None
            st.rerun()

    # 标杆状态
    current_profile = st.session_state.benchmark_profile
    if current_profile:
        if 'positive' in current_profile: st.success("✅ 双向标杆：已激活")
        else: st.success("✅ 单向标杆：已激活")
        if st.button("清除标杆", use_container_width=True):
            st.session_state.benchmark_profile = None; st.rerun()
    
    # 算法参数
    with st.expander("⚙️ 基础算法参数", expanded=False):
        p_width = st.slider("处理分辨率", 256, 1024, 512, 128)
        k_num = st.slider("色彩聚类数", 2, 8, 5)
        st.caption("阈值微调")
        t_diag = st.slider("对角线判定", 0.1, 0.5, 0.3)
        t_sym_blur = st.slider("对称模糊K", 1, 51, 31, 2)
        ref_tex = st.slider("纹理基准", 10.0, 100.0, 50.0)
        t_clarity = st.slider("高光/清晰阈值", 0.5, 0.9, 0.7)
    
    # 权重容差
    with st.expander("⚖️ 评分权重配置", expanded=False):
        dims_geo = [
            ('comp_balance_score', '感知平衡'), ('comp_layout_score', '构图匹配'), 
            ('comp_negative_space_score', '呼吸感'), ('comp_visual_flow_score', '视线引导'),
            ('comp_visual_order_score', '视觉秩序')
        ]
        dims_color = [
            ('color_saturation', '饱和度'), ('color_brightness', '亮度'), 
            ('color_warmth', '暖色调'), ('color_contrast', '对比度'), 
            ('color_clarity', '清晰度'), ('color_harmony', '和谐度')
        ]
        dims_text = [
            ('text_alignment_score', '排版对齐'), ('text_hierarchy_score', '层级性'),
            ('text_content_ratio', '内容占比'), ('fg_text_legibility', '易读性(含对比)')
        ]
        dims_content = [('fg_color_diff', '主体色差'), ('fg_area_diff', '主体占比'), ('fg_texture_diff', '纹理差异')]
        
        loaded_weights = current_profile.get('weights', {}) if current_profile else {}
        
        final_weights = {}
        
        def render_sliders(col, category_name, dims):
            col.caption(f"**{category_name}**")
            for k, label in dims:
                default_val = float(loaded_weights.get(k, 1.0)) 
                key = f"w_{k}"
                if key not in st.session_state: st.session_state[key] = default_val
                final_weights[k] = col.slider(label, 0.0, 5.0, step=0.1, key=key)
        
        c_w1, c_w2 = st.columns(2)
        render_sliders(c_w1, "📐 构图/秩序", dims_geo)
        render_sliders(c_w1, "🎨 色彩", dims_color)
        render_sliders(c_w2, "🅰️ 文字排版", dims_text)
        render_sliders(c_w2, "🌗 图底", dims_content)

    # [Updated] 将自定义提示词打包进 Config
    config = {
        'process_width': p_width, 'seg_kmeans_k': k_num, 'comp_diag_slope': t_diag,
        'comp_sym_blur_k': t_sym_blur, 'fg_tex_norm': ref_tex, 'color_clarity_thresh': t_clarity,
        'comp_thirds_slope': 0.2, 'comp_sym_tolerance': 120.0, 'text_score_thresh': 60.0,
        'weights': final_weights, 
        'analysis_prompt': st.session_state.analysis_prompt    # 传递给后端
    }

# 初始化引擎
@st.cache_resource
def get_engine(api_key, endpoint, _version="v25.0_vlm_rec"):
    return OmniVisualEngine(vlm_api_key=api_key, vlm_endpoint=endpoint)

engine = get_engine(vlm_key, vlm_endpoint)

# ==========================================
# 4. 核心工具函数
# ==========================================

def make_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def calculate_dual_score(data, profile, bm_manager):
    is_dual = 'positive' in profile and isinstance(profile['positive'], dict)
    if is_dual:
        res_pos = bm_manager.score_against_benchmark(data, profile['positive'])
        score_pos = res_pos['total_score']
        score_neg = 0
        if 'negative' in profile and profile['negative']:
            res_neg = bm_manager.score_against_benchmark(data, profile['negative'])
            score_neg = res_neg['total_score'] 
        penalty_factor = 0.4 
        final_score = max(0, min(100, score_pos - (score_neg * penalty_factor)))
        if final_score >= 90: rating = "S (卓越)"
        elif final_score >= 80: rating = "A (优秀)"
        elif final_score >= 70: rating = "B (良好)"
        elif final_score >= 60: rating = "C (合格)"
        else: rating = "D (不合格)"
        return {
            'total_score': final_score, 'rating_level': rating, 'mode': '双向标杆',
            'details': res_pos['details'], 'score_breakdown': {'pos': score_pos, 'neg': score_neg}
        }
    else:
        res = bm_manager.score_against_benchmark(data, profile)
        res['mode'] = '单向标杆'
        res['score_breakdown'] = None
        return res

def normalize_values(source, is_profile=False):
    def get(k): 
        val = source.get(k, {}).get('target', 0) if is_profile else getattr(source, k, 0)
        return float(val) if val is not None else 0.0
    
    return [
        get('comp_balance_score'), get('comp_layout_score'), get('comp_negative_space_score'), 
        get('comp_visual_flow_score'), get('comp_visual_order_score'),
        
        get('color_warmth')*100, get('color_saturation')*100, get('color_brightness')*100, min(100, (get('color_contrast')/0.3)*100), get('color_clarity')*100, get('color_harmony'),
        
        get('text_alignment_score'), get('text_hierarchy_score'), min(100, get('text_content_ratio') * 2), get('fg_text_legibility'),
        
        get('fg_area_diff')*100, min(100, get('fg_color_diff')), get('fg_texture_diff')*100
    ]

def plot_aesthetic_manifold(manifold, current_vector=None):
    if not manifold or not manifold.is_fitted: return None
    vis_data = manifold.get_visualization_data()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vis_data['x'], y=vis_data['y'], mode='markers',
        marker=dict(size=8, color='rgba(46, 204, 113, 0.6)', line=dict(width=1, color='DarkSlateGrey')),
        text=vis_data['filenames'], name='标杆正向样本'
    ))
    if current_vector is not None:
        _, _, curr_coord = manifold.evaluate(current_vector)
        fig.add_trace(go.Scatter(
            x=[curr_coord[0]], y=[curr_coord[1]], mode='markers',
            marker=dict(size=15, color='red', symbol='star'), name='当前图片'
        ))
    fig.update_layout(
        title="🌌 动态美学流形 (PCA 2D Projection)", xaxis_title="Feature Dim 1", yaxis_title="Feature Dim 2",
        showlegend=True, height=400, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="rgba(240,242,246,0.5)"
    )
    return fig

# ==========================================
# 5. 批量处理逻辑
# ==========================================
def run_batch_process(files, cfg, need_zip, profile=None):
    st.session_state.processing = True
    st.session_state.batch_logs = []
    ALL_DIMS_MAPPING = [
        ('comp_balance_score', '构图_感知平衡'), ('comp_layout_score', '构图_模板匹配'),
        ('comp_negative_space_score', '构图_呼吸感'), ('comp_visual_flow_score', '构图_视线引导'),
        ('comp_visual_order_score', '构图_视觉秩序'),
        ('color_saturation', '色彩_饱和度'), ('color_brightness', '色彩_亮度'),
        ('color_warmth', '色彩_暖色调'), ('color_contrast', '色彩_对比度'),
        ('color_clarity', '色彩_清晰度'), ('color_harmony', '色彩_和谐度'),
        ('text_alignment_score', '文字_排版对齐'), ('text_hierarchy_score', '文字_层级性'),
        ('text_content_ratio', '文字_内容占比'), ('fg_text_legibility', '文字_易读性'), 
        ('fg_color_diff', '图底_色差'), ('fg_area_diff', '图底_占比'), ('fg_texture_diff', '图底_纹理差')
    ]
    rows = []; diff_rows = []; raw_json_list = []
    zip_buffer = io.BytesIO() if need_zip else None
    zf = zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) if need_zip else None
    bm_manager = BenchmarkManager() if profile else None
    total = len(files); progress_bar = st.progress(0); status_text = st.empty()
    
    for idx, f in enumerate(files):
        try:
            status_text.text(f"Processing {idx+1}/{total}: {f.name}")
            f.seek(0); f_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8); img_bgr = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None: continue
            d = engine.analyze(img_bgr, config=cfg)
            if profile:
                if 'positive' in profile: res = calculate_dual_score(d, profile, bm_manager); target_dict = profile['positive']
                else: res = bm_manager.score_against_benchmark(d, profile); target_dict = profile
                final_score = res['total_score']; final_rating = res['rating_level']; mode_str = f"标杆 ({res.get('mode','默认')})"
            else:
                rep = AestheticDiagnostician.generate_report(d, config=cfg)
                final_score = rep['total_score']; final_rating = rep['rating_level']; mode_str = "通用模式"; target_dict = {}
            base_info = {"文件名": f.name, "综合得分": final_score, "评级": final_rating, "模式": mode_str}
            row_data = base_info.copy(); diff_data = base_info.copy()
            for key, label in ALL_DIMS_MAPPING:
                val = getattr(d, key, 0) or 0
                if key == 'fg_text_legibility' and not getattr(d, 'fg_text_present', False): val = 0
                row_data[label] = round(val, 2)
                if profile and key in target_dict: t_val = target_dict[key].get('target', 0); diff_data[f"Δ_{label}"] = round(val - t_val, 2)
                else: diff_data[f"Δ_{label}"] = 0
            if hasattr(d, 'kobayashi_tags') and d.kobayashi_tags: row_data['印象标签'] = ", ".join(d.kobayashi_tags)
            rows.append(row_data); diff_rows.append(diff_data)
            raw_obj = {k: make_serializable(getattr(d, k)) for k, _ in ALL_DIMS_MAPPING}; raw_obj['filename'] = f.name; raw_json_list.append(raw_obj)
            if zf:
                vis_map = {
                    'v_balance': 'vis_saliency_heatmap', 'v_layout': 'vis_layout_template', 
                    'v_flow': 'vis_visual_flow', 'v_order': 'vis_visual_order',
                    'v_sat': 'vis_saturation', 'v_bri': 'vis_brightness', 'v_text_leg': 'vis_text_analysis', 'v_text_lay': 'vis_text_design',
                    'v_col_harm': 'vis_color_harmony'
                }
                base_name = f.name.rsplit('.', 1)[0]
                for excel_key, attr_name in vis_map.items():
                    img_data = getattr(d, attr_name, None)
                    if img_data is not None:
                        if hasattr(img_data, 'dtype') and img_data.dtype != np.uint8: img_data = img_data.astype(np.uint8)
                        if len(img_data.shape) == 2: img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
                        _, buf = cv2.imencode('.png', cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)); zf.writestr(f"diagnostics/{base_name}_{excel_key}.png", buf.tobytes())
            del d; del img_bgr; 
            if idx % 5 == 0: gc.collect()
        except Exception as e: st.session_state.batch_logs.append(f"Error {f.name}: {e}")
        progress_bar.progress((idx + 1) / total)
    if zf: zf.close()
    st.session_state.batch_df = pd.DataFrame(rows); st.session_state.batch_diff_df = pd.DataFrame(diff_rows); st.session_state.batch_raw_json = raw_json_list; st.session_state.batch_zip = zip_buffer.getvalue() if need_zip else None; st.session_state.processing = False; gc.collect()

# ==========================================
# 6. 主界面逻辑 (按模式)
# ==========================================

# --- 模式 1: 批量工厂 ---
if mode == "📦 批量工厂": 
    st.title("📦 批量处理中心 (Placeholder)")
    st.info("批量功能已折叠...")

# --- 模式 2: 单图诊断 ---
elif mode == "📸 单图诊断":
    st.title("🧿 单图深度诊断")
    uploaded_file = st.file_uploader("上传图片", type=['jpg','png','jpeg'])
    
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        with st.spinner("AI 正在进行全维度扫描..."):
            try:
                data = engine.analyze(img_bgr, config=config)
                
                final_score = 0; final_rating = "N/A"; mode_display = "通用"; penalty_info = {}; bench_details = {}; is_bench = False
                
                # Hybrid Scoring Logic
                if st.session_state.aesthetic_manifold and st.session_state.benchmark_profile:
                    is_bench = True
                    trainer = BenchmarkTrainer()
                    res = trainer.calculate_hybrid_score(data, st.session_state.aesthetic_manifold, st.session_state.benchmark_profile)
                    final_score = res['total_score']; final_rating = res['rating_level']; mode_display = "混合 (Hybrid)"
                    penalty_info = res['components']; bench_details = res['rule_details']
                elif st.session_state.benchmark_profile:
                    is_bench = True
                    bm = BenchmarkManager()
                    res = bm.score_against_benchmark(data, st.session_state.benchmark_profile)
                    final_score = res['total_score']; final_rating = res['rating_level']; bench_details = res['details']; mode_display = "规则标杆"
                else:
                    rep = AestheticDiagnostician.generate_report(data, config=config)
                    final_score = rep['total_score']; final_rating = rep['rating_level']

            except Exception as e:
                st.error(f"Analysis Failed: {e}"); st.stop()

        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.image(image_pil, use_container_width=True)
            st.metric("🏆 综合得分", f"{final_score:.1f}", delta=f"{final_rating} ({mode_display})")
            
            with st.expander("📊 评分构成", expanded=True):
                sc1, sc2, sc3 = st.columns(3)
                if penalty_info:
                    sc1.metric("流形分", f"{penalty_info['manifold_score']:.1f}", help="基于k-NN (60%)")
                    sc2.metric("规则分", f"{penalty_info['rule_score']:.1f}", help="基于标杆统计 (40%)")
                    pen = penalty_info.get('penalty_factor', 0)
                    sc3.metric("技术惩罚", f"-{pen*100:.0f}%", delta_color="inverse" if pen > 0 else "normal")
                    if pen > 0:
                        for r in penalty_info.get('penalty_reasons', []): st.caption(f"⚠️ {r}")
                elif mode_display == "规则标杆":
                    sc1.metric("流形分", "N/A")
                    sc2.metric("规则分", f"{final_score:.1f}")
                    sc3.metric("技术惩罚", "0%")
                else:
                    sc1.metric("流形分", "N/A")
                    sc2.metric("规则分", "N/A")
                    sc3.metric("技术惩罚", "N/A")

            if st.session_state.aesthetic_manifold:
                st.divider()
                st.plotly_chart(plot_aesthetic_manifold(st.session_state.aesthetic_manifold, data.to_feature_vector()), use_container_width=True)
                if penalty_info and penalty_info.get('neighbors'):
                    st.caption(f"相似标杆: {', '.join(penalty_info['neighbors'][:3])}")
            
            # [Updated] 展示 VLM 识别状态，替代原有的点评
            if vlm_key:
                with st.expander("🤖 VLM 识别状态", expanded=False):
                    st.success("VLM 调用成功")
                    if getattr(data, 'fg_text_present', False):
                        st.write(f"📝 **识别文字**: {data.fg_text_content}")
                    else:
                        st.write("📝 **识别文字**: 无")
            else:
                st.warning("VLM 未配置，识别功能受限")

            # Smart Cards
            def smart_card(col, label, key, unit="", multiplier=1.0):
                raw_val = getattr(data, key, 0) or 0
                if is_bench and key in bench_details:
                    item = bench_details[key]; score = item['score']; actual = item['actual'] * multiplier; target = item['target'] * multiplier
                    col.metric(label, f"{score:.0f}", f"{actual:.1f} / {target:.1f}{unit}")
                else: col.metric(label, f"{raw_val*multiplier:.1f}{unit}")

            st.divider(); st.caption("🎨 色彩 (6项)")
            r1c1, r1c2, r1c3 = st.columns(3)
            smart_card(r1c1, "饱和度", "color_saturation", "%", 100); smart_card(r1c2, "亮度", "color_brightness", "%", 100); smart_card(r1c3, "暖色", "color_warmth", "%", 100)
            r2c1, r2c2, r2c3 = st.columns(3)
            smart_card(r2c1, "对比度", "color_contrast", "", 1.0); smart_card(r2c2, "清晰度", "color_clarity", "%", 100); smart_card(r2c3, "和谐度", "color_harmony", "", 1.0)

            st.divider(); st.caption("📐 构图与视觉秩序 (5项)")
            g_r1_1, g_r1_2 = st.columns(2); smart_card(g_r1_1, "感知平衡", "comp_balance_score")
            lay = getattr(data, 'comp_layout_type', 'N/A'); smart_card(g_r1_2, f"构图 ({lay})", "comp_layout_score")
            r4c1, r4c2, r4c3 = st.columns(3); smart_card(r4c1, "呼吸感", "comp_negative_space_score"); smart_card(r4c2, "视线引导", "comp_visual_flow_score"); smart_card(r4c3, "视觉秩序", "comp_visual_order_score")

            st.divider(); st.caption("🅰️ 文字排版 (4项)")
            t_r1_1, t_r1_2 = st.columns(2); smart_card(t_r1_1, "排版对齐", "text_alignment_score"); smart_card(t_r1_2, "层级性", "text_hierarchy_score")
            t_r2_1, t_r2_2 = st.columns(2); 
            smart_card(t_r2_1, "内容占比", "text_content_ratio", "%"); 
            if getattr(data, 'fg_text_present', False): 
                smart_card(t_r2_2, "易读性", "fg_text_legibility")
            else: 
                t_r2_2.metric("易读性", "N/A", "无显著文字")

            st.divider(); st.caption("🌗 图底与信息 (3项)")
            f_r1_1, f_r1_2, f_r1_3 = st.columns(3)
            smart_card(f_r1_1, "主体色差", "fg_color_diff")
            smart_card(f_r1_2, "主体占比", "fg_area_diff", "%", 100)
            smart_card(f_r1_3, "纹理差异", "fg_texture_diff", "%", 100)
            
        with c2:
            st.subheader("📊 维度雷达 (18核心)")
            cats = ['感知平衡','构图匹配','呼吸感','视线引导', '视觉秩序', '暖色','饱和','亮度','对比','清晰','和谐', '排版对齐', '层级', '内容比', '易读', '占比', '色差', '纹理']
            vals = normalize_values(data, False); fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself', name='当前图片', line_color='#3498db'))
            if is_bench:
                bench_vals = normalize_values(st.session_state.benchmark_profile['positive'] if 'positive' in st.session_state.benchmark_profile else st.session_state.benchmark_profile, True)
                fig.add_trace(go.Scatterpolar(r=bench_vals, theta=cats, fill='toself', name='标杆基准', line_color='#2ecc71', opacity=0.4))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=350, margin=dict(t=20, b=20, l=40, r=40))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🔎 诊断图谱")
            t_comp, t_color, t_fg, t_text = st.tabs(["📐 构图/秩序", "🎨 色彩", "🌗 图底", "🅰️ 排版"])
            with t_comp:
                if data.vis_saliency_heatmap is not None: st.image(data.vis_saliency_heatmap, caption="视觉平衡热力图", use_container_width=True)
                if getattr(data, 'vis_layout_dict', None):
                    opts = [f"{k} ({v['score']:.0f})" for k, v in sorted(data.vis_layout_dict.items(), key=lambda x:x[1]['score'], reverse=True)]
                    sel = st.radio("构图模板", opts, horizontal=True)
                    if sel: st.image(data.vis_layout_dict[sel.split(' (')[0]]['vis'], use_container_width=True)
                c3, c4 = st.columns(2)
                if data.vis_visual_flow is not None: c3.image(data.vis_visual_flow, caption="视线引导分析", use_container_width=True)
                if data.vis_visual_order is not None: c4.image(data.vis_visual_order, caption="视觉秩序 (角度熵)", use_container_width=True)
            with t_color:
                c1, c2 = st.columns(2)
                if data.vis_warmth is not None: c1.image(data.vis_warmth, caption="冷暖分布", use_container_width=True)
                if data.vis_color_harmony is not None: c2.image(data.vis_color_harmony, caption="和谐色轮", use_container_width=True)
                c3, c4 = st.columns(2)
                if data.vis_brightness is not None: c3.image(data.vis_brightness, caption="亮度(J)分布", use_container_width=True)
                if data.vis_clarity is not None: c4.image(data.vis_clarity, caption="清晰度/高光", use_container_width=True)
            with t_fg:
                c1, c2 = st.columns(2)
                if data.vis_mask is not None: c1.image(data.vis_mask, caption="智能分割 (VLM+SAM)", use_container_width=True)
                if data.vis_color_contrast is not None: c2.image(data.vis_color_contrast, caption="色彩对比", use_container_width=True)
                c3, c4 = st.columns(2)
                if data.vis_edge_composite is not None: c3.image(data.vis_edge_composite, caption="纹理对比", use_container_width=True)
            with t_text:
                c1, c2 = st.columns(2)
                if data.vis_text_analysis is not None: c1.image(data.vis_text_analysis, caption="易读性分析 (VLM OCR)", use_container_width=True)
                if data.vis_text_design is not None: c2.image(data.vis_text_design, caption="排版分析", use_container_width=True)

# --- 模式 3: 建立标杆 --- 
elif mode == "🏆 建立标杆":
    st.title("🏆 建立行业视觉标杆 (流形版)")
    with st.expander("📂 加载配置", expanded=False):
        uploaded_profile = st.file_uploader("上传 benchmark_profile.json", type=["json"], key="profile_loader")
        if uploaded_profile and st.button("确认加载"):
            loaded = json.load(uploaded_profile)
            st.session_state.benchmark_profile = loaded
            if 'manifold_bytes' in loaded and loaded['manifold_bytes']:
                try:
                    st.session_state.aesthetic_manifold = AestheticManifold.load(base64.b64decode(loaded['manifold_bytes']))
                    st.success("✅ 配置与模型已加载！")
                except: st.warning("模型加载失败")
            time.sleep(1); st.rerun()

    c1, c2 = st.columns(2)
    files_high = c1.file_uploader("High (正向样本 - 构建流形)", accept_multiple_files=True)
    files_low = c2.file_uploader("Low (负向样本 - 辅助权重)", accept_multiple_files=True)
    
    if files_high and st.button("🚀 开始训练"):
        trainer = BenchmarkTrainer()
        gc.collect()
        with st.spinner("正在提取 18D 特征向量并构建流形空间..."):
            try:
                profile, dist_data, stats, manifold = trainer.train(files_high, files_low, config)
                st.session_state.benchmark_profile = profile
                st.session_state.aesthetic_manifold = manifold
                st.success(f"完成！正向:{stats['pos_count']}, 负向:{stats['neg_count']}")
                
                st.plotly_chart(plot_aesthetic_manifold(manifold), use_container_width=True)
                
                with st.expander("📈 特征分布", expanded=True):
                    tp, tn = st.tabs(["🟢 正向", "🔴 负向"])
                    with tp:
                        fig = go.Figure()
                        for k, v in dist_data['pos'].items(): fig.add_trace(go.Box(y=v, name=k))
                        st.plotly_chart(fig, use_container_width=True)
                    with tn:
                        if dist_data['neg']:
                            fig = go.Figure()
                            for k, v in dist_data['neg'].items(): fig.add_trace(go.Box(y=v, name=k))
                            st.plotly_chart(fig, use_container_width=True)
                        else: st.info("未上传负向样本。")

                json_str = json.dumps(profile, default=make_serializable, indent=4)
                st.download_button("📦 下载配置", json_str, "benchmark_profile.json", "application/json", type="primary")
            except Exception as e: st.error(f"Error: {e}")