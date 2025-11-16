import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate
from utils.scoring import compute_bleu, compute_efc

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="Premium MT Evaluation Dashboard", layout="wide")

# --------------------------------------------------------------
# LOAD LOTTIE ANIMATION
# --------------------------------------------------------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None
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
        "accent1": "#4facfe",
        "accent2": "#43e97b",
        "accent3": "#fa709a"
    },
    "Light": {
        "bg": "linear-gradient(135deg, #ffffff, #f0f7ff)",
        "card_bg": "rgba(0,0,0,0.05)",
        "text": "#101624",
        "muted": "#444444",
        "accent1": "#0b78d1",
        "accent2": "#16a34a",
        "accent3": "#d63384"
    }
}

theme = st.sidebar.selectbox("Choose Theme", ["Dark", "Light"])
C = THEMES[theme]

# --------------------------------------------------------------
# CSS STYLING
# --------------------------------------------------------------
st.markdown(f"""
<style>
body {{
    background: {C['bg']};
    color: {C['text']};
}}

.kpi-glass {{
  background: {C['card_bg']};
  backdrop-filter: blur(10px);
  padding: 25px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.15);
  box-shadow: 0 8px 28px rgba(0,0,0,0.35);
  transition: transform 0.3s;
}}
.kpi-glass:hover {{
  transform: translateY(-5px);
}}

.kpi-circle {{
    width: 150px; height: 150px;
    border-radius:50%;
    background: conic-gradient(var(--color) calc(var(--value) * 1%), #333 0%);
    display:flex; align-items:center; justify-content:center;
    margin:auto; box-shadow:0 0 20px var(--color-glow);
}}
.kpi-circle-inner {{
    width:110px; height:110px;
    background:#0d0d0d;
    border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:26px; font-weight:700; color:white;
}}
.metric-bar {{
    height: 18px; border-radius: 10px; background: #333;
    overflow:hidden; position:relative;
}}
.metric-bar-fill {{
    height:100%; border-radius:10px;
    animation:fillBar 1.8s ease forwards;
}}
@keyframes fillBar {{
    from {{ width:0%; }}
    to {{ width:var(--width); }}
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# HEADER WITH LOTTIE
# --------------------------------------------------------------
lottie = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
colA, colB = st.columns([1,4])

with colA:
    if lottie:
        st_lottie(lottie, height=130)

with colB:
    st.markdown(f"<h1 style='color:{C['text']}'>🚀 Next-Level Animated MT Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{C['muted']}'>Premium UI • Animated KPIs • 3D Charts • Radar • Tabs</p>", unsafe_allow_html=True)

st.write("---")

# --------------------------------------------------------------
# INPUT
# --------------------------------------------------------------
text = st.text_area("Enter text to evaluate:", height=120)

# --------------------------------------------------------------
# METRIC FUNCTION
# --------------------------------------------------------------
def get_metrics(out, src):
    bleu = compute_bleu(src, out)
    efc = compute_efc(src, out)
    halluc = round(1 - efc, 3)
    semantic = round((bleu + efc) / 2, 3)
    return {
        "BLEU": bleu,
        "EFC": efc,
        "Hallucination": halluc,
        "Semantic": semantic
    }

# --------------------------------------------------------------
# EVALUATE BUTTON
# --------------------------------------------------------------
if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Please enter text.")
    else:
        # RUN MODELS (HIDDEN)
        out_b = baseline_translate(text)
        out_e = eact_translate(text)
        out_r = rgcld_translate(text)

        mB = get_metrics(out_b, text)
        mE = get_metrics(out_e, text)
        mR = get_metrics(out_r, text)

        # --------------------------------------------------------------
        # TABS FOR METRICS
        # --------------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💎 KPI Rings",
            "📈 3D Metrics",
            "🧭 Radar Comparison",
            "📉 Advanced Metrics",
            "📊 Comparison Table"
        ])

        # --------------------------------------------------------------
        # TAB 1 — KPI RINGS (Animated)
        # --------------------------------------------------------------
        with tab1:
            c1, c2, c3 = st.columns(3)
            for (name, model, accent), col in zip(
                [("Baseline", mB, C['accent1']),
                 ("EACT", mE, C['accent2']),
                 ("RG-CLD", mR, C['accent3'])],
                [c1, c2, c3]
            ):
                bleu_val = model["BLEU"]
                col.markdown(f"""
                <div class="kpi-glass">
                    <h3 style='text-align:center;color:{accent}'>{name}</h3>
                    <div class="kpi-circle" style="--value:{bleu_val*100}; --color:{accent}; --color-glow:{accent}55;">
                        <div class="kpi-circle-inner">{bleu_val}</div>
                    </div>
                    <p style='text-align:center;margin-top:10px;color:{C['muted']}'>BLEU Score</p>
                </div>
                """, unsafe_allow_html=True)

        # --------------------------------------------------------------
        # TAB 2 — 3D Plotly Chart
        # --------------------------------------------------------------
        with tab2:
            fig3d = go.Figure(data=[go.Scatter3d(
                x=[mB["BLEU"], mE["BLEU"], mR["BLEU"]],
                y=[mB["EFC"], mE["EFC"], mR["EFC"]],
                z=[mB["Semantic"], mE["Semantic"], mR["Semantic"]],
                mode='markers+text',
                text=["Baseline", "EACT", "RG-CLD"],
                marker=dict(size=8, color=[0,1,2], colorscale='Plotly3')
            )])
            fig3d.update_layout(
                scene=dict(
                    xaxis_title='BLEU',
                    yaxis_title='EFC',
                    zaxis_title='Semantic'
                ),
                height=600,
            )
            st.plotly_chart(fig3d, use_container_width=True)

        # --------------------------------------------------------------
        # TAB 3 — Animated Radar (Plotly)
        # --------------------------------------------------------------
        with tab3:
            categories = ["BLEU", "EFC", "Hallucination", "Semantic"]

            figRadar = go.Figure()

            figRadar.add_trace(go.Scatterpolar(
                r=[mB[c] for c in categories],
                theta=categories,
                fill='toself',
                name="Baseline"
            ))
            figRadar.add_trace(go.Scatterpolar(
                r=[mE[c] for c in categories],
                theta=categories,
                fill='toself',
                name="EACT"
            ))
            figRadar.add_trace(go.Scatterpolar(
                r=[mR[c] for c in categories],
                theta=categories,
                fill='toself',
                name="RG-CLD"
            ))

            figRadar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=600
            )

            st.plotly_chart(figRadar, use_container_width=True)

        # --------------------------------------------------------------
        # TAB 4 — Advanced Metrics (Animated Bars)
        # --------------------------------------------------------------
        with tab4:
            colA, colB = st.columns(2)

            # Left side — Hallucination
            with colA:
                st.markdown("### Hallucination Rate (Lower = Better)")
                for name, m, color in [
                    ("Baseline", mB, "#ff4e50"),
                    ("EACT", mE, "#ffa500"),
                    ("RG-CLD", mR, "#ff2a68")
                ]:
                    st.markdown(f"""
                    <p><b>{name}</b>: {m['Hallucination']}</p>
                    <div class="metric-bar">
                        <div class="metric-bar-fill" style="--width:{m['Hallucination']*100}%; background:{color};"></div>
                    </div><br>
                    """, unsafe_allow_html=True)

            # Right side — Semantic Similarity
            with colB:
                st.markdown("### Semantic Similarity (Higher = Better)")
                for name, m, color in [
                    ("Baseline", mB, "#30cfd0"),
                    ("EACT", mE, "#6a5acd"),
                    ("RG-CLD", mR, "#4facfe")
                ]:
                    st.markdown(f"""
                    <p><b>{name}</b>: {m['Semantic']}</p>
                    <div class="metric-bar">
                        <div class="metric-bar-fill" style="--width:{m['Semantic']*100}%; background:{color};"></div>
                    </div><br>
                    """, unsafe_allow_html=True)

        # --------------------------------------------------------------
        # TAB 5 — Comparison Table
        # --------------------------------------------------------------
        with tab5:
            st.write("### Model Comparison Matrix")
            st.table({
                "Model": ["Baseline", "EACT", "RG-CLD"],
                "BLEU": [mB["BLEU"], mE["BLEU"], mR["BLEU"]],
                "EFC": [mB["EFC"], mE["EFC"], mR["EFC"]],
                "Hallucination": [mB["Hallucination"], mE["Hallucination"], mR["Hallucination"]],
                "Semantic": [mB["Semantic"], mE["Semantic"], mR["Semantic"]],
            })
