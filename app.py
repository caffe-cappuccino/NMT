# app.py
import streamlit as st
import time
import random
import plotly.graph_objects as go
import numpy as np
import requests
from streamlit_lottie import st_lottie

# import your model wrappers and scoring utils (must exist)
from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate
from utils.scoring import compute_bleu, compute_efc

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Next-Level MT Dashboard — Animated", layout="wide")

# ---------------------------
# Helper: load lottie
# ---------------------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

# ---------------------------
# THEME / UI COLORS
# ---------------------------
THEMES = {
    "Dark": {
        "bg": "linear-gradient(135deg, #0b1020, #0f1724)",
        "card_bg": "rgba(255,255,255,0.06)",
        "text": "#FFFFFF",
        "muted": "#BFC8D6",
        "accent1": "#4facfe",
        "accent2": "#43e97b",
        "accent3": "#fa709a"
    },
    "Light": {
        "bg": "linear-gradient(135deg, #ffffff, #f0f6ff)",
        "card_bg": "rgba(0,0,0,0.04)",
        "text": "#0b1220",
        "muted": "#444444",
        "accent1": "#0b78d1",
        "accent2": "#16a34a",
        "accent3": "#d63384"
    }
}

# default theme
chosen_theme = st.sidebar.selectbox("AI Theme (switcher)", ["Dark", "Light"], index=0)
C = THEMES[chosen_theme]

# inject basic CSS for glass + theme
st.markdown(f"""
<style>
:root{{--bg: {C['bg']}; --card-bg:{C['card_bg']}; --text:{C['text']}; --muted:{C['muted']}; --accent1:{C['accent1']}; --accent2:{C['accent2']}; --accent3:{C['accent3']};}}
body {{ background: var(--bg); color: var(--text); }}
div.stButton > button {{ background: linear-gradient(90deg, var(--accent1), var(--accent2)); color: #fff; border: none; padding: 8px 16px; border-radius:8px; }}
.kpi-glass {{
  background: var(--card-bg);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border-radius: 14px;
  padding: 18px;
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 8px 30px rgba(2,6,23,0.35);
}}
.small-muted {{ color: var(--muted); font-size:13px; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header (with Lottie)
# ---------------------------
col_head1, col_head2 = st.columns([1,4])
with col_head1:
    lottie_anim = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
    if lottie_anim:
        st_lottie(lottie_anim, height=140)
with col_head2:
    st.markdown(f"<h1 style='margin:0; color: {C['text']};'>🚀 Next-Level Animated MT Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>AI theme: <b>{chosen_theme}</b> · 3D charts · animated radar · Lottie · streaming metrics</div>", unsafe_allow_html=True)

st.write("---")

# ---------------------------
# Input controls
# ---------------------------
left, right = st.columns([3,1])
with left:
    text = st.text_area("Enter text to evaluate (models run but translations are hidden):", height=130)
    model_choice = st.selectbox("Primary language pair (placeholder)", ["English → Hindi"], index=0)
with right:
    st.markdown("**Streaming Controls**")
    stream_toggle = st.button("Start Streaming")
    stream_steps = st.slider("Stream updates (steps)", min_value=5, max_value=40, value=12, step=1)
    stream_delay = st.slider("Update interval (sec)", min_value=0.1, max_value=1.5, value=0.4, step=0.1)

# ---------------------------
# Utility to compute metrics
# ---------------------------
def compute_all_metrics(src_text, out_text):
    bleu = compute_bleu(src_text, out_text)
    efc = compute_efc(src_text, out_text)
    halluc = round(1 - efc, 3)
    semantic = round((bleu + efc) / 2, 3)
    return {"BLEU": bleu, "EFC": efc, "Hallucination": halluc, "Semantic": semantic}

# ---------------------------
# Prepare session state for streaming history
# ---------------------------
if "history" not in st.session_state:
    st.session_state["history"] = {
        "steps": [],
        "baseline": [],
        "eact": [],
        "rgcld": []
    }

# ---------------------------
# Evaluate button (single-run)
# ---------------------------
if st.button("Run Evaluation Now"):
    if not text.strip():
        st.error("Please enter text.")
    else:
        # run models (hidden)
        b_out = baseline_translate(text)
        e_out = eact_translate(text)
        r_out = rgcld_translate(text)

        # metrics
        m_b = compute_all_metrics(text, b_out)
        m_e = compute_all_metrics(text, e_out)
        m_r = compute_all_metrics(text, r_out)

        # show premium KPI cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="kpi-glass">
                    <h3 style='margin:0; color:{C['accent1']}'>Baseline</h3>
                    <div style='font-size:22px; font-weight:700; color:{C['text']};'>{m_b['BLEU']}</div>
                    <div class='small-muted'>BLEU</div>
                    <div style='height:8px'></div>
                    <div style='font-size:18px; font-weight:600; color:{C['text']};'>{m_b['EFC']}</div>
                    <div class='small-muted'>EFC</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="kpi-glass">
                    <h3 style='margin:0; color:{C['accent2']}'>EACT</h3>
                    <div style='font-size:22px; font-weight:700; color:{C['text']};'>{m_e['BLEU']}</div>
                    <div class='small-muted'>BLEU</div>
                    <div style='height:8px'></div>
                    <div style='font-size:18px; font-weight:600; color:{C['text']};'>{m_e['EFC']}</div>
                    <div class='small-muted'>EFC</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="kpi-glass">
                    <h3 style='margin:0; color:{C['accent3']}'>RG-CLD</h3>
                    <div style='font-size:22px; font-weight:700; color:{C['text']};'>{m_r['BLEU']}</div>
                    <div class='small-muted'>BLEU</div>
                    <div style='height:8px'></div>
                    <div style='font-size:18px; font-weight:600; color:{C['text']};'>{m_r['EFC']}</div>
                    <div class='small-muted'>EFC</div>
                </div>
            """, unsafe_allow_html=True)

        # quick static 3D plot of metric trio (BLEU / EFC / Semantic)
        st.markdown("### 3D Metric Space (BLEU, EFC, Semantic) — static snapshot")
        fig3d = go.Figure(data=[go.Scatter3d(
            x=[m_b['BLEU'], m_e['BLEU'], m_r['BLEU']],
            y=[m_b['EFC'], m_e['EFC'], m_r['EFC']],
            z=[m_b['Semantic'], m_e['Semantic'], m_r['Semantic']],
            mode='markers+text',
            marker=dict(size=8, color=[0,1,2], colorscale='Viridis'),
            text=["Baseline","EACT","RG-CLD"],
            textposition="top center"
        )])
        fig3d.update_layout(scene=dict(xaxis_title='BLEU', yaxis_title='EFC', zaxis_title='Semantic'))
        st.plotly_chart(fig3d, use_container_width=True)

        # radar (comparison)
        categories = ["BLEU","EFC","Hallucination","Semantic"]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=[m_b[c] for c in categories], theta=categories, fill='toself', name='Baseline'))
        fig_radar.add_trace(go.Scatterpolar(r=[m_e[c] for c in categories], theta=categories, fill='toself', name='EACT'))
        fig_radar.add_trace(go.Scatterpolar(r=[m_r[c] for c in categories], theta=categories, fill='toself', name='RG-CLD'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------
# Streaming mode: simulated real-time metric updates
# ---------------------------
if stream_toggle:
    if not text.strip():
        st.error("Please enter text to stream metrics for.")
    else:
        # initialize run history
        st.session_state.history = {"steps": [], "baseline": [], "eact": [], "rgcld": []}
        placeholder_3d = st.empty()
        placeholder_radar = st.empty()
        placeholder_table = st.empty()
        placeholder_kpis = st.empty()
        # small warmup run: compute base metrics to seed
        b_out = baseline_translate(text)
        e_out = eact_translate(text)
        r_out = rgcld_translate(text)
        base_b = compute_all = lambda: None  # placeholder to satisfy linter

        for step in range(1, stream_steps + 1):
            # simulate some realistic jitter around base metrics by re-running models
            b_out = baseline_translate(text)
            e_out = eact_translate(text)
            r_out = rgcld_translate(text)

            mb = compute_all_metrics(text, b_out)
            me = compute_all_metrics(text, e_out)
            mr = compute_all_metrics(text, r_out)

            # add small random drift to simulate live updates
            def jitter(val):
                jittered = max(0.0, min(1.0, round(val + random.uniform(-0.03, 0.03), 3)))
                return jittered

            mb_j = {k: jitter(v) for k, v in mb.items()}
            me_j = {k: jitter(v) for k, v in me.items()}
            mr_j = {k: jitter(v) for k, v in mr.items()}

            st.session_state.history["steps"].append(step)
            st.session_state.history["baseline"].append(mb_j)
            st.session_state.history["eact"].append(me_j)
            st.session_state.history["rgcld"].append(mr_j)

            # KPI small cards live (render)
            with placeholder_kpis.container():
                st.markdown("<div style='display:flex; gap:14px;'>", unsafe_allow_html=True)
                for name, data, accent in [
                    ("Baseline", mb_j, C['accent1']),
                    ("EACT", me_j, C['accent2']),
                    ("RG-CLD", mr_j, C['accent3'])
                ]:
                    st.markdown(f"""
                        <div class='kpi-glass' style='flex:1; padding:14px;'>
                            <div style='display:flex; justify-content:space-between;'>
                                <div>
                                    <div style='font-size:14px; color:{C['muted']};'>{name}</div>
                                    <div style='font-weight:700; font-size:20px; color:{C['text']};'>{data['BLEU']}</div>
                                    <div style='font-size:12px; color:{C['muted']};'>BLEU</div>
                                </div>
                                <div style='text-align:right;'>
                                    <div style='font-weight:700; font-size:20px; color:{accent};'>{data['EFC']}</div>
                                    <div style='font-size:12px; color:{C['muted']};'>EFC</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # 3D time-series: BLEU vs time for all models
            steps = st.session_state.history["steps"]
            bleu_b_ts = [d["BLEU"] for d in st.session_state.history["baseline"]]
            bleu_e_ts = [d["BLEU"] for d in st.session_state.history["eact"]]
            bleu_r_ts = [d["BLEU"] for d in st.session_state.history["rgcld"]]

            fig3d_live = go.Figure()
            fig3d_live.add_trace(go.Scatter3d(x=steps, y=bleu_b_ts, z=[0]*len(steps), mode='lines+markers', name='Baseline', marker=dict(size=4, color=C['accent1'])))
            fig3d_live.add_trace(go.Scatter3d(x=steps, y=bleu_e_ts, z=[1]*len(steps), mode='lines+markers', name='EACT', marker=dict(size=4, color=C['accent2'])))
            fig3d_live.add_trace(go.Scatter3d(x=steps, y=bleu_r_ts, z=[2]*len(steps), mode='lines+markers', name='RG-CLD', marker=dict(size=4, color=C['accent3'])))
            fig3d_live.update_layout(scene=dict(xaxis_title='Time step', yaxis_title='BLEU', zaxis_title='Model (index)'), margin=dict(l=0,r=0,b=0,t=30))
            placeholder_3d.plotly_chart(fig3d_live, use_container_width=True)

            # radar live (latest snapshot)
            latest_b = st.session_state.history["baseline"][-1]
            latest_e = st.session_state.history["eact"][-1]
            latest_r = st.session_state.history["rgcld"][-1]
            cats = ["BLEU","EFC","Hallucination","Semantic"]
            fig_radar_live = go.Figure()
            fig_radar_live.add_trace(go.Scatterpolar(r=[latest_b[c] for c in cats], theta=cats, fill='toself', name='Baseline'))
            fig_radar_live.add_trace(go.Scatterpolar(r=[latest_e[c] for c in cats], theta=cats, fill='toself', name='EACT'))
            fig_radar_live.add_trace(go.Scatterpolar(r=[latest_r[c] for c in cats], theta=cats, fill='toself', name='RG-CLD'))
            fig_radar_live.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, margin=dict(l=0,r=0,b=0,t=30))
            placeholder_radar.plotly_chart(fig_radar_live, use_container_width=True)

            # data table
            placeholder_table.table({
                "Model": ["Baseline", "EACT", "RG-CLD"],
                "BLEU": [latest_b["BLEU"], latest_e["BLEU"], latest_r["BLEU"]],
                "EFC": [latest_b["EFC"], latest_e["EFC"], latest_r["EFC"]],
                "Hallucination": [latest_b["Hallucination"], latest_e["Hallucination"], latest_r["Hallucination"]],
                "Semantic": [latest_b["Semantic"], latest_e["Semantic"], latest_r["Semantic"]]
            })

            time.sleep(stream_delay)

        st.success("Streaming complete ✅")

# ---------------------------
# Footer / tips
# ---------------------------
st.write("---")
st.markdown("<div class='small-muted'>Tip: Use the Start Streaming button to visualize live metric drift. Use Run Evaluation Now for a static snapshot.</div>", unsafe_allow_html=True)
