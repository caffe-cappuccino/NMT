# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
import time

# Import model wrappers + scoring utils
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

loading_animation = load_lottie("https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json")
header_animation = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")

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
# HEADER
# --------------------------------------------------------------
col1, col2 = st.columns([1,4])
with col1:
    if header_animation:
        st_lottie(header_animation, height=120)

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

    halluc = 1 - efc
    semantic = (bleu + efc) / 2

    return {
        "BLEU": float(np.clip(bleu, 0, 1)),
        "EFC": float(np.clip(efc, 0, 1)),
        "Hallucination": float(np.clip(halluc, 0, 1)),
        "Semantic": float(np.clip(semantic, 0, 1))
    }

# --------------------------------------------------------------
# RUN BUTTON + INLINE LOADING ANIMATION
# --------------------------------------------------------------

loading_placeholder = st.empty()   # << animation appears HERE

if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Please enter text first.")
    else:

        # Show loading animation BELOW button
        with loading_placeholder:
            if loading_animation:
                st_lottie(loading_animation, height=180)

        time.sleep(1.5)

        # Run models
        out_b = baseline_translate(text)
        out_e = eact_translate(text)
        out_r = rgcld_translate(text)

        mB = get_metrics(text, out_b)
        mE = get_metrics(text, out_e)
        mR = get_metrics(text, out_r)

        # Remove animation
        loading_placeholder.empty()

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
        # TAB 1 – KPI RINGS
        # ----------------------------------------------------------
        with tab1:
            c1, c2, c3 = st.columns(3)
            for (title, metrics, accent), col in zip(
                [("Baseline", mB, C['acc1']),
                 ("EACT", mE, C['acc2']),
                 ("RG-CLD", mR, C['acc3'])],
                [c1, c2, c3]
            ):
                bval = metrics["BLEU"]
                col.markdown(f"""
                <div class="kpi-glass">
                  <h3 style='text-align:center; color:{accent}'>{title}</h3>
                  <div class="kpi-circle" style="--value:{bval*100}; --color:{accent}; --color-glow:{accent}55;">
                    <div class="kpi-circle-inner">{bval}</div>
                  </div>
                  <p style='text-align:center; color:{C['muted']}; margin-top:8px;'>BLEU Score</p>
                </div>
                """, unsafe_allow_html=True)

        # ----------------------------------------------------------
        # TAB 2 – 3D BLEU–EFC LINE
        # ----------------------------------------------------------
        with tab2:
            st.markdown("### 📈 3D BLEU–EFC Trajectory (Gradient, Dotted, Mesh)")

            X = [mB["BLEU"], mE["BLEU"], mR["BLEU"]]
            Y = [mB["EFC"], mE["EFC"], mR["EFC"]]
            Z = [0, 0.5, 1]
            labels = ["Baseline", "EACT", "RG-CLD"]

            t_original = np.linspace(0, 1, len(X))
            t_fine = np.linspace(0, 1, 200)

            xs = np.interp(t_fine, t_original, X)
            ys = np.interp(t_fine, t_original, Y)
            zs = np.interp(t_fine, t_original, Z)

            fig = go.Figure()

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(width=8, color=xs, colorscale='Turbo', dash='dot'),
                hoverinfo='none'
            ))

            fig.add_trace(go.Scatter3d(
                x=X, y=Y, z=Z,
                mode='markers+text',
                marker=dict(size=10, color=[C['acc1'], C['acc2'], C['acc3']],
                            line=dict(width=2, color='white')),
                text=labels,
                textposition='top center'
            ))

            xx, yy = np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))
            fig.add_trace(go.Surface(
                x=xx, y=yy, z=np.zeros_like(xx),
                opacity=0.1, showscale=False,
                colorscale=[[0,'rgba(130,130,130,0.15)'],[1,'rgba(200,200,200,0.02)']]
            ))

            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='BLEU', range=[0,1]),
                    yaxis=dict(title='EFC',  range=[0,1]),
                    zaxis=dict(title='Depth', showticklabels=False)
                ),
                height=650
            )

            st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------------
        # TAB 3 – RADAR
        # ----------------------------------------------------------
        with tab3:
            cats = ["BLEU", "EFC", "Hallucination", "Semantic"]
            figR = go.Figure()

            figR.add_trace(go.Scatterpolar(r=[mB[c] for c in cats], theta=cats, fill='toself', name="Baseline"))
            figR.add_trace(go.Scatterpolar(r=[mE[c] for c in cats], theta=cats, fill='toself', name="EACT"))
            figR.add_trace(go.Scatterpolar(r=[mR[c] for c in cats], theta=cats, fill='toself', name="RG-CLD"))

            figR.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                height=620
            )

            st.plotly_chart(figR, use_container_width=True)

        # ----------------------------------------------------------
        # TAB 4 – ADVANCED METRICS
        # ----------------------------------------------------------
        with tab4:
            colA, colB = st.columns(2)

            with colA:
                st.markdown("### Hallucination Rate")
                for name, metrics, color in [
                    ("Baseline", mB, "#ff4e50"),
                    ("EACT", mE, "#ffa600"),
                    ("RG-CLD", mR, "#ff2a68")
                ]:
                    val = metrics["Hallucination"]
                    st.markdown(f"<b>{name}</b>: {val}", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-bar'><div class='metric-bar-fill' style='width:{val*100}%; background:{color};'></div></div><br>", unsafe_allow_html=True)

            with colB:
                st.markdown("### Semantic Similarity")
                for name, metrics, color in [
                    ("Baseline", mB, "#30cfd0"),
                    ("EACT", mE, "#6a5acd"),
                    ("RG-CLD", mR, "#4facfe")
                ]:
                    val = metrics["Semantic"]
                    st.markdown(f"<b>{name}</b>: {val}", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-bar'><div class='metric-bar-fill' style='width:{val*100}%; background:{color};'></div></div><br>", unsafe_allow_html=True)

        # ----------------------------------------------------------
        # TAB 5 – TABLE
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
