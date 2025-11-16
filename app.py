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
# Page config
# --------------------------------------------------------------
st.set_page_config(page_title="Neural Translation Evaluation & Insights Platform", layout="wide")

# --------------------------------------------------------------
# Lottie loader
# --------------------------------------------------------------
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

loading_animation = load_lottie("https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json")

# --------------------------------------------------------------
# Themes
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
# CSS + FIXED TYPEWRITER WIDTH (cursor stops at last letter)
# --------------------------------------------------------------
st.markdown(f"""
<style>
body {{
    background:{C['bg']};
    color:{C['text']};
}}

@keyframes typing {{
    from {{ width: 0 }}
    to {{ width: 96ch; }}  /* EXACT width of caption text */
}}

@keyframes blink {{
    50% {{ border-color: transparent; }}
}}

.typewriter {{
    display: inline-block;
    white-space: nowrap;
    overflow: hidden;
    border-right: 3px solid {C['text']};
    width: 96ch;                     /* CURSOR stops EXACTLY at last character */
    animation: typing 3.5s steps(96, end), blink .75s step-end infinite;
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
    margin:auto;box-shadow:0 0 18px var(--color-glow);
}}
.kpi-circle-inner {{
    width:100px;height:100px;
    background: rgba(0,0,0,0.55);
    border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:22px;font-weight:700;color:white;
}}
.metric-bar {{ height:16px;border-radius:10px;background:#333; overflow:hidden; }}
.metric-bar-fill {{ height:100%; border-radius:10px; transition: width 1.6s ease; }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# HEADER + FIXED TYPEWRITER CAPTION
# --------------------------------------------------------------
lottie_header = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
col1, col2 = st.columns([1,4])
with col1:
    if lottie_header:
        st_lottie(lottie_header, height=120)
with col2:
    st.markdown(
        f"<h1 style='color:{C['text']}; margin:0;'>Neural Translation Evaluation & Insights Platform</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='typewriter'>AI-driven analytics for benchmarking translation quality, coherence, and linguistic fidelity.</div>",
        unsafe_allow_html=True
    )

st.write("---")

# --------------------------------------------------------------
# INPUT
# --------------------------------------------------------------
text = st.text_area("Enter text to evaluate:", height=140)

# --------------------------------------------------------------
# METRICS
# --------------------------------------------------------------
def get_metrics(src_text, out_text):
    bleu = compute_bleu(src_text, out_text)
    efc = compute_efc(src_text, out_text)
    halluc = round(1 - efc, 3)
    semantic = round((bleu + efc) / 2, 3)
    return {
        "BLEU": float(np.clip(bleu, 0, 1)),
        "EFC": float(np.clip(efc, 0, 1)),
        "Hallucination": float(np.clip(halluc, 0, 1)),
        "Semantic": float(np.clip(semantic, 0, 1))
    }

# --------------------------------------------------------------
# RUN BUTTON (NO “Running evaluation…” TEXT)
# --------------------------------------------------------------
if st.button("Run Evaluation"):

    if not text.strip():
        st.error("Please enter text.")

    else:
        # Clean loading animation only
        loader = st.empty()
        with loader:
            st_lottie(loading_animation, height=200)

        time.sleep(2)

        # Run models
        out_b = baseline_translate(text)
        out_e = eact_translate(text)
        out_r = rgcld_translate(text)

        # Compute metrics
        mB = get_metrics(text, out_b)
        mE = get_metrics(text, out_e)
        mR = get_metrics(text, out_r)

        loader.empty()

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
        # TAB 1 — KPI Rings
        # ----------------------------------------------------------
        with tab1:
            c1, c2, c3 = st.columns(3)
            for (title, mat, accent), col in zip(
                [("Baseline", mB, C['acc1']),
                 ("EACT", mE, C['acc2']),
                 ("RG-CLD", mR, C['acc3'])],
                [c1, c2, c3]
            ):
                val = mat["BLEU"]
                col.markdown(f"""
                <div class="kpi-glass">
                  <h3 style='text-align:center; color:{accent}'>{title}</h3>
                  <div class="kpi-circle" style="--value:{val*100}; --color:{accent}; --color-glow:{accent}55;">
                    <div class="kpi-circle-inner">{val}</div>
                  </div>
                  <p style='text-align:center; color:{C['muted']}'>BLEU Score</p>
                </div>
                """, unsafe_allow_html=True)

        # ----------------------------------------------------------
        # TAB 2 — 3D BLEU–EFC
        # ----------------------------------------------------------
        with tab2:
            st.markdown("### 📈 3D BLEU–EFC Trajectory")

            X = [mB["BLEU"], mE["BLEU"], mR["BLEU"]]
            Y = [mB["EFC"], mE["EFC"], mR["EFC"]]
            Z = [0.0, 0.5, 1.0]

            t = np.linspace(0, 1, 200)
            xs = np.interp(t, [0, 0.5, 1], X)
            ys = np.interp(t, [0, 0.5, 1], Y)
            zs = np.interp(t, [0, 0.5, 1], Z)

            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(width=8, color=xs, colorscale="Turbo", dash="dot")
            ))
            fig.add_trace(go.Scatter3d(
                x=X, y=Y, z=Z,
                mode="markers+text",
                marker=dict(size=10, color=[C['acc1'], C['acc2'], C['acc3']], line=dict(width=2, color="white")),
                text=["Baseline", "EACT", "RG-CLD"],
                textposition="top center"
            ))

            fig.update_layout(height=620, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------------
        # TAB 3 — Radar
        # ----------------------------------------------------------
        with tab3:
            cats = ["BLEU", "EFC", "Hallucination", "Semantic"]
            figR = go.Figure()
            figR.add_trace(go.Scatterpolar(r=[mB[c] for c in cats], theta=cats, fill='toself'))
            figR.add_trace(go.Scatterpolar(r=[mE[c] for c in cats], theta=cats, fill='toself'))
            figR.add_trace(go.Scatterpolar(r=[mR[c] for c in cats], theta=cats, fill='toself'))
            figR.update_layout(height=620)
            st.plotly_chart(figR, use_container_width=True)

        # ----------------------------------------------------------
        # TAB 4 — Bars
        # ----------------------------------------------------------
        with tab4:
            cA, cB = st.columns(2)
            with cA:
                st.markdown("### Hallucination Rate")
                for name, m, clr in [("Baseline",mB,"#ff4e50"),("EACT",mE,"#ffa600"),("RG-CLD",mR,"#ff2a68")]:
                    v = m["Hallucination"]
                    st.markdown(f"<b>{name}</b>: {v}", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-bar'><div class='metric-bar-fill' style='width:{v*100}%;background:{clr}'></div></div><br>", unsafe_allow_html=True)

            with cB:
                st.markdown("### Semantic Similarity")
                for name, m, clr in [("Baseline",mB,"#30cfd0"),("EACT",mE,"#6a5acd"),("RG-CLD",mR,"#4facfe")]:
                    v = m["Semantic"]
                    st.markdown(f"<b>{name}</b>: {v}", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-bar'><div class='metric-bar-fill' style='width:{v*100}%;background:{clr}'></div></div><br>", unsafe_allow_html=True)

        # ----------------------------------------------------------
        # TAB 5 — Table
        # ----------------------------------------------------------
        with tab5:
            st.write("### Model Comparison Matrix")
            st.table({
                "Model": ["Baseline", "EACT", "RG-CLD"],
                "BLEU": [mB["BLEU"], mE["BLEU"], mR["BLEU"]],
                "EFC": [mB["EFC"], mE["EFC"], mR["EFC"]],
                "Hallucination": [mB["Hallucination"], mE["Hallucination"], mR["Hallucination"]],
                "Semantic": [mB["Semantic"], mE["Semantic"], mR["Semantic"]],
            })
