# ===============================================================
#  PREMIUM NEXT-LEVEL MT DASHBOARD — FULL APP.PY
# ===============================================================

import streamlit as st
import time
import random
import plotly.graph_objects as go
import numpy as np
import requests
from streamlit_lottie import st_lottie

# Your model files
from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate

# Your metric utils
from utils.scoring import compute_bleu, compute_efc

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="Ultra-Premium MT Dashboard", layout="wide")

# --------------------------------------------------------------
# LOAD LOTTIE
# --------------------------------------------------------------
def load_lottie(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

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
  padding:25px;
  border-radius:14px;
  border:1px solid rgba(255,255,255,0.15);
  box-shadow:0 8px 28px rgba(0,0,0,0.35);
}}
.kpi-circle {{
    width:150px;height:150px;border-radius:50%;
    background: conic-gradient(var(--color) calc(var(--value) * 1%), #333 0%);
    display:flex;align-items:center;justify-content:center;
    margin:auto;box-shadow:0 0 20px var(--color-glow);
}}
.kpi-circle-inner {{
    width:110px;height:110px;
    background:#0d0d0d;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:26px;font-weight:700;color:white;
}}
.metric-bar {{
    height:18px;border-radius:10px;background:#333;
    overflow:hidden;
}}
.metric-bar-fill {{
    height:100%;border-radius:10px;
    animation:fillBar 1.7s ease forwards;
}}
@keyframes fillBar {{
    from {{width:0%;}}
    to {{width:var(--width);}}
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# HEADER
# --------------------------------------------------------------
lottie = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
c1, c2 = st.columns([1,4])
with c1:
    if lottie:
        st_lottie(lottie, height=140)
with c2:
    st.markdown(f"<h1 style='color:{C['text']}'>🚀 Ultra-Premium Animated MT Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{C['muted']}'>Now with gradient 3D line, mesh planes, transparent axes & dotted curves</p>", unsafe_allow_html=True)
st.write("---")

# --------------------------------------------------------------
# INPUT
# --------------------------------------------------------------
text = st.text_area("Enter text to evaluate:", height=120)

# --------------------------------------------------------------
# METRIC FUNCTION
# --------------------------------------------------------------
def metrics(src, out):
    bleu = compute_bleu(src, out)
    efc = compute_efc(src, out)
    halluc = round(1 - efc, 3)
    semantic = round((bleu + efc) / 2, 3)
    return {"BLEU": bleu, "EFC": efc, "Hallucination": halluc, "Semantic": semantic}

# --------------------------------------------------------------
# RUN EVALUATION
# --------------------------------------------------------------
if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Please enter text first.")
    else:
        # Run models (hidden)
        out_b = baseline_translate(text)
        out_e = eact_translate(text)
        out_r = rgcld_translate(text)

        mB = metrics(text, out_b)
        mE = metrics(text, out_e)
        mR = metrics(text, out_r)

        # ----------------------------------------------------------
        # TABS
        # ----------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💎 KPI Rings",
            "📈 3D Gradient Line",
            "🧭 Radar Chart",
            "📉 Advanced Metrics",
            "📊 Table"
        ])

        # ----------------------------------------------------------
        # TAB 1 — KPI RINGS
        # ----------------------------------------------------------
        with tab1:
            cA, cB, cC = st.columns(3)
            for (title, data, acc), col in zip(
                [("Baseline", mB, C['acc1']),
                 ("EACT", mE, C['acc2']),
                 ("RG-CLD", mR, C['acc3'])],
                [cA, cB, cC]
            ):
                b = data["BLEU"]
                col.markdown(f"""
                <div class="kpi-glass">
                    <h3 style="text-align:center;color:{acc}">{title}</h3>
                    <div class="kpi-circle" style="--value:{b*100}; --color:{acc}; --color-glow:{acc}66;">
                        <div class="kpi-circle-inner">{b}</div>
                    </div>
                    <p style='text-align:center;color:{C["muted"]}'>BLEU Score</p>
                </div>
                """, unsafe_allow_html=True)

        # ----------------------------------------------------------
        # TAB 2 — 3D GRADIENT LINE + DOTTED + TRANSPARENT PLANES + MESH
        # ----------------------------------------------------------
        with tab2:
            st.markdown("### 📈 Premium 3D Metric Space with Gradient + Mesh + Dotted Line")

            x = [mB["BLEU"], mE["BLEU"], mR["BLEU"]]
            y = [mB["EFC"], mE["EFC"], mR["EFC"]]
            z = [mB["Semantic"], mE["Semantic"], mR["Semantic"]]

            labels = ["Baseline", "EACT", "RG-CLD"]

            # Create smooth curve segments for gradient
            xs = np.linspace(x[0], x[-1], 50)
            ys = np.linspace(y[0], y[-1], 50)
            zs = np.linspace(z[0], z[-1], 50)

            fig3d = go.Figure()

            # Gradient dotted line
            fig3d.add_trace(go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='lines',
                line=dict(
                    width=6,
                    color=np.linspace(0,1,50),
                    colorscale='Rainbow',
                    dash='dot'  # dotted pattern
                ),
                name="Trajectory"
            ))

            # Add markers
            fig3d.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers+text',
                marker=dict(size=8, color=[0,1,2], colorscale="Rainbow"),
                text=labels,
                textposition="top center",
                name="Models"
            ))

            # Transparent axis planes
            fig3d.update_scenes(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0.1)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0.05)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0.05)")
            )

            # Mesh surfaces — bounding cube-like guidance planes
            xx, yy = np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))
            zz = np.zeros_like(xx)
            fig3d.add_trace(go.Surface(
                x=xx, y=yy, z=zz,
                opacity=0.15,
                showscale=False,
                colorscale="Blues",
                name="Boundary Surface"
            ))

            fig3d.update_layout(
                height=600,
                scene=dict(
                    xaxis_title='BLEU',
                    yaxis_title='EFC',
                    zaxis_title='Semantic',
                )
            )

            st.plotly_chart(fig3d, use_container_width=True)

        # ----------------------------------------------------------
        # TAB 3 — RADAR CHART
        # ----------------------------------------------------------
        with tab3:
            cats = ["BLEU", "EFC", "Hallucination", "Semantic"]

            figR = go.Figure()
            figR.add_trace(go.Scatterpolar(
                r=[mB[c] for c in cats], theta=cats, fill='toself', name="Baseline"))
            figR.add_trace(go.Scatterpolar(
                r=[mE[c] for c in cats], theta=cats, fill='toself', name="EACT"))
            figR.add_trace(go.Scatterpolar(
                r=[mR[c] for c in cats], theta=cats, fill='toself', name="RG-CLD"))

            figR.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                height=600
            )
            st.plotly_chart(figR, use_container_width=True)

        # ----------------------------------------------------------
        # TAB 4 — ADVANCED METRICS (Animated Bars)
        # ----------------------------------------------------------
        with tab4:
            colX, colY = st.columns(2)

            with colX:
                st.markdown("### Hallucination Rate")
                for name, m, clr in [
                    ("Baseline", mB, "#ff4e50"),
                    ("EACT", mE, "#ffa600"),
                    ("RG-CLD", mR, "#ff2a68")
                ]:
                    st.markdown(f"""
                    <p><b>{name}</b>: {m['Hallucination']}</p>
                    <div class="metric-bar">
                        <div class="metric-bar-fill" style="--width:{m['Hallucination']*100}%; background:{clr};"></div>
                    </div><br>
                    """, unsafe_allow_html=True)

            with colY:
                st.markdown("### Semantic Similarity")
                for name, m, clr in [
                    ("Baseline", mB, "#30cfd0"),
                    ("EACT", mE, "#6a5acd"),
                    ("RG-CLD", mR, "#4facfe")
                ]:
                    st.markdown(f"""
                    <p><b>{name}</b>: {m['Semantic']}</p>
                    <div class="metric-bar">
                        <div class="metric-bar-fill" style="--width:{m['Semantic']*100}%; background:{clr};"></div>
                    </div><br>
                    """, unsafe_allow_html=True)

        # ----------------------------------------------------------
        # TAB 5 — TABLE
        # ----------------------------------------------------------
        with tab5:
            st.write("### Comparison Table")
            st.table({
                "Model": ["Baseline", "EACT", "RG-CLD"],
                "BLEU": [mB["BLEU"], mE["BLEU"], mR["BLEU"]],
                "EFC": [mB["EFC"], mE["EFC"], mR["EFC"]],
                "Hallucination": [mB["Hallucination"], mE["Hallucination"], mR["Hallucination"]],
                "Semantic": [mB["Semantic"], mE["Semantic"], mR["Semantic"]],
            })
