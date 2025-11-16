# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

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
# CSS
# --------------------------------------------------------------
st.markdown(f"""
<style>
body {{
    background:{C['bg']};
    color:{C['text']};
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
# Header with Lottie & NEW Caption
# --------------------------------------------------------------
lottie = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
col1, col2 = st.columns([1,4])
with col1:
    if lottie:
        st_lottie(lottie, height=120)

with col2:
    st.markdown(
        f"<h1 style='color:{C['text']}; margin:0;'>Neural Translation Evaluation & Insights Platform</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='color:{C['muted']}; font-size:18px; margin-top:4px;'>"
        f"AI-driven analytics for benchmarking translation quality, coherence, and linguistic fidelity."
        f"</div>",
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

    # clip to [0,1]
    return {
        "BLEU": float(np.clip(bleu, 0, 1)),
        "EFC": float(np.clip(efc, 0, 1)),
        "Hallucination": float(np.clip(halluc, 0, 1)),
        "Semantic": float(np.clip(semantic, 0, 1))
    }

# --------------------------------------------------------------
# RUN BUTTON
# --------------------------------------------------------------
if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Please enter text to evaluate.")
    else:
        # Run models
        out_b = baseline_translate(text)
        out_e = eact_translate(text)
        out_r = rgcld_translate(text)

        mB = get_metrics(text, out_b)
        mE = get_metrics(text, out_e)
        mR = get_metrics(text, out_r)

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
                bval = mat["BLEU"]
                col.markdown(f"""
                <div class="kpi-glass">
                  <h3 style='text-align:center; color:{accent}; margin:0'>{title}</h3>
                  <div class="kpi-circle" style="--value:{bval*100}; --color:{accent}; --color-glow:{accent}55;">
                    <div class="kpi-circle-inner">{bval}</div>
                  </div>
                  <p style='text-align:center; color:{C['muted']}; margin-top:8px;'>BLEU Score</p>
                </div>
                """, unsafe_allow_html=True)

        # ----------------------------------------------------------
        # TAB 2 — Enhanced 3D BLEU–EFC Line
        # ----------------------------------------------------------
        with tab2:
            st.markdown("### 📈 3D BLEU–EFC Trajectory (Gradient + Mesh + Dotted Line)")

            X = [mB["BLEU"], mE["BLEU"], mR["BLEU"]]
            Y = [mB["EFC"],  mE["EFC"],  mR["EFC"]]
            Z = [0.0, 0.5, 1.0]
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
                line=dict(
                    width=8,
                    color=xs,
                    colorscale='Turbo',
                    dash='dot'
                ),
                hoverinfo='none',
                name='Trajectory'
            ))

            fig.add_trace(go.Scatter3d(
                x=X, y=Y, z=Z,
                mode='markers+text',
                marker=dict(size=9, color=[C['acc1'], C['acc2'], C['acc3']],
                            line=dict(width=2, color='white')),
                text=labels,
                textposition='top center',
                name='Models'
            ))

            xx, yy = np.meshgrid(np.linspace(0,1,6), np.linspace(0,1,6))
            zz0 = np.zeros_like(xx)
            fig.add_trace(go.Surface(
                x=xx, y=yy, z=zz0,
                showscale=False, opacity=0.10,
                colorscale=[[0, 'rgba(100,100,100,0.12)'], [1, 'rgba(200,200,200,0.02)']]
            ))

            zz1 = 0.2 + 0.1 * (xx + yy)
            fig.add_trace(go.Surface(
                x=xx, y=yy, z=zz1,
                showscale=False, opacity=0.06,
                colorscale='Blues'
            ))

            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='BLEU', range=[0,1], backgroundcolor='rgba(0,0,0,0)'),
                    yaxis=dict(title='EFC',  range=[0,1], backgroundcolor='rgba(0,0,0,0)'),
                    zaxis=dict(title='Depth', showticklabels=False),
                ),
                margin=dict(l=0,r=0,t=50,b=0),
                height=640,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------------
        # TAB 3 — Radar Chart
        # ----------------------------------------------------------
        with tab3:
            cats = ["BLEU", "EFC", "Hallucination", "Semantic"]
            figR = go.Figure()
            figR.add_trace(go.Scatterpolar(r=[mB[c] for c in cats], theta=cats, fill='toself', name="Baseline"))
            figR.add_trace(go.Scatterpolar(r=[mE[c] for c in cats], theta=cats, fill='toself', name="EACT"))
            figR.add_trace(go.Scatterpolar(r=[mR[c] for c in cats], theta=cats, fill='toself', name="RG-CLD"))
            figR.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=620)
            st.plotly_chart(figR, use_container_width=True)

        # ----------------------------------------------------------
        # TAB 4 — Advanced Metrics
        # ----------------------------------------------------------
        with tab4:
            colA, colB = st.columns(2)

            with colA:
                st.markdown("### Hallucination Rate (Lower = Better)")
                for name, metrics, color in [
                    ("Baseline", mB, "#ff4e50"),
                    ("EACT", mE, "#ffa600"),
                    ("RG-CLD", mR, "#ff2a68")
                ]:
                    val = metrics["Hallucination"]
                    st.markdown(f"<b>{name}</b>: {val}", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-bar'><div class='metric-bar-fill' style='width:{val*100}%; background:{color};'></div></div><br>", unsafe_allow_html=True)

            with colB:
                st.markdown("### Semantic Similarity (Higher = Better)")
                for name, metrics, color in [
                    ("Baseline", mB, "#30cfd0"),
                    ("EACT", mE, "#6a5acd"),
                    ("RG-CLD", mR, "#4facfe")
                ]:
                    val = metrics["Semantic"]
                    st.markdown(f"<b>{name}</b>: {val}", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-bar'><div class='metric-bar-fill' style='width:{val*100}%; background:{color};'></div></div><br>", unsafe_allow_html=True)

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
