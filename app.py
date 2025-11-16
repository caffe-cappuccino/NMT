# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
import time

# Import your model wrappers and scoring utils
from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate
from utils.scoring import compute_bleu, compute_efc

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="Neural Translation Evaluation & Insights Platform", layout="wide")

# --------------------------------------------------------------
# LOTTIE LOADER
# --------------------------------------------------------------
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

header_animation = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
loading_animation = load_lottie("https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json")

# --------------------------------------------------------------
# SESSION STATE (MANDATORY for visible loading)
# --------------------------------------------------------------
if "run_eval" not in st.session_state:
    st.session_state.run_eval = False
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
if "metrics" not in st.session_state:
    st.session_state.metrics = None

# --------------------------------------------------------------
# THEMES
# --------------------------------------------------------------
THEMES = {
    "Dark": {
        "bg": "linear-gradient(135deg, #0b1020, #131b2f)",
        "card_bg": "rgba(255,255,255,0.07)",
        "text": "#FFFFFF",
        "muted": "#BFC8D6",
        "acc1": "#4facfe",
        "acc2": "#43e97b",
        "acc3": "#fa709a"
    },
    "Light": {
        "bg": "linear-gradient(135deg, #ffffff, #f0f7ff)",
        "card_bg": "rgba(0,0,0,0.05)",
        "text": "#101624",
        "muted": "#444444",
        "acc1": "#0b78d1",
        "acc2": "#16a34a",
        "acc3": "#d63384"
    }
}

theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])
C = THEMES[theme]

# --------------------------------------------------------------
# CSS + TYPEWRITER
# --------------------------------------------------------------
st.markdown(f"""
<style>

body {{
    background:{C['bg']};
    color:{C['text']};
}}

@keyframes typing {{
    from {{ width: 0 }}
    to {{ width: 100% }}
}}

@keyframes blink {{
    50% {{ border-color: transparent; }}
}}

.typewriter {{
    width: 100%;
    overflow: hidden;
    white-space: nowrap;
    border-right: 3px solid {C['text']};
    animation: typing 3.5s steps(60, end), blink .75s step-end infinite;
    font-size: 18px;
    color:{C['muted']};
}}

.kpi-glass {{
  background:{C['card_bg']};
  backdrop-filter:blur(10px);
  padding:20px;
  border-radius:12px;
  border:1px solid rgba(255,255,255,0.12);
  box-shadow:0 8px 24px rgba(0,0,0,0.35);
}}
.kpi-circle {{
    width:140px;height:140px;border-radius:50%;
    background: conic-gradient(var(--color) calc(var(--value) * 1%), #333 0%);
    display:flex;align-items:center;justify-content:center;
    margin:auto;
}}
.kpi-circle-inner {{
    width:100px;height:100px;
    background: rgba(0,0,0,0.55);
    border-radius:50%;
    font-size:22px;font-weight:700;color:white;
    display:flex;align-items:center;justify-content:center;
}}

.metric-bar {{ height:16px;border-radius:10px;background:#333; overflow:hidden; }}
.metric-bar-fill {{ height:100%; border-radius:10px; transition: width 1.6s ease; }}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# HEADER
# --------------------------------------------------------------
col1, col2 = st.columns([1,4])
with col1:
    if header_animation:
        st_lottie(header_animation, height=120)

with col2:
    st.markdown(
        f"<h1 style='color:{C['text']}; margin:0;'>Neural Translation Evaluation & Insights Platform</h1>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div class='typewriter'>AI-driven analytics for benchmarking translation quality, coherence, and linguistic fidelity.</div>",
        unsafe_allow_html=True)

st.write("---")

# --------------------------------------------------------------
# INPUT
# --------------------------------------------------------------
text = st.text_area("Enter text to evaluate:", height=140)

# --------------------------------------------------------------
# METRIC FUNCTION
# --------------------------------------------------------------
def get_metrics(src, out):
    bleu = compute_bleu(src, out)
    efc = compute_efc(src, out)
    halluc = 1 - efc
    semantic = (bleu + efc)/2
    return {
        "BLEU": float(np.clip(bleu, 0, 1)),
        "EFC": float(np.clip(efc, 0, 1)),
        "Hallucination": float(np.clip(halluc, 0, 1)),
        "Semantic": float(np.clip(semantic, 0, 1)),
    }

# --------------------------------------------------------------
# RUN BUTTON
# --------------------------------------------------------------
if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Enter text first.")
    else:
        st.session_state.run_eval = True
        st.session_state.results_ready = False
        st.experimental_rerun()   # <<< FORCE UI UPDATE

# --------------------------------------------------------------
# PHASE 1 — SHOW LOADING ANIMATION
# --------------------------------------------------------------
if st.session_state.run_eval and not st.session_state.results_ready:

    st_lottie(loading_animation, height=180)

    # Give animation time to appear
    time.sleep(1)

    # Run heavy model code
    out_b = baseline_translate(text)
    out_e = eact_translate(text)
    out_r = rgcld_translate(text)

    mB = get_metrics(text, out_b)
    mE = get_metrics(text, out_e)
    mR = get_metrics(text, out_r)

    st.session_state.metrics = (mB, mE, mR)

    st.session_state.results_ready = True
    st.session_state.run_eval = False

    st.experimental_rerun()   # <<< GO TO PHASE 2


# --------------------------------------------------------------
# PHASE 2 — SHOW RESULTS
# --------------------------------------------------------------
if st.session_state.results_ready:

    mB, mE, mR = st.session_state.metrics

    # ----------------------------------------------------------
    # TABS
    # ----------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💎 KPI Rings",
        "📈 3D BLEU–EFC Line",
        "🧭 Radar Comparison",
        "📉 Advanced Metrics",
        "📊 Comparison Table"
    ])

    # ----------------------------------------------------------
    # KPI RINGS
    # ----------------------------------------------------------
    with tab1:
        c1, c2, c3 = st.columns(3)
        for (name, m, accent), col in zip(
            [("Baseline", mB, C['acc1']),
             ("EACT", mE, C['acc2']),
             ("RG-CLD", mR, C['acc3'])],
            [c1, c2, c3]
        ):
            v = m["BLEU"]
            col.markdown(f"""
            <div class="kpi-glass">
                <h3 style='text-align:center; color:{accent};'>{name}</h3>
                <div class="kpi-circle" style="--value:{v*100}; --color:{accent}; --color-glow:{accent}55;">
                    <div class="kpi-circle-inner">{v}</div>
                </div>
                <p style='text-align:center; color:{C['muted']}; margin-top:8px;'>BLEU Score</p>
            </div>
            """, unsafe_allow_html=True)

    # ----------------------------------------------------------
    # OTHER TABS (unchanged—you already have them)
    # ----------------------------------------------------------
    # HERE goes your 3D line, radar, bars, and table (same code)

