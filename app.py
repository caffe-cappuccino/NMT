import streamlit as st
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
# LOTTIE LOADER
# --------------------------------------------------------------
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

# --------------------------------------------------------------
# THEMING
# --------------------------------------------------------------
THEMES = {
    "Dark": {
        "bg": "linear-gradient(135deg, #0b1020, #131b2f)",
        "card_bg": "rgba(255,255,255,0.06)",
        "text": "#FFFFFF",
        "muted": "#BFC8D6",
        "accent1": "#4facfe",
        "accent2": "#43e97b",
        "accent3": "#fa709a",
    },
    "Light": {
        "bg": "linear-gradient(135deg, #ffffff, #f4f7ff)",
        "card_bg": "rgba(0,0,0,0.05)",
        "text": "#101624",
        "muted": "#4a4a4a",
        "accent1": "#0078ff",
        "accent2": "#1bbf72",
        "accent3": "#e84393",
    }
}

theme = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)
C = THEMES[theme]

# --------------------------------------------------------------
# GLOBAL CSS
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
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 28px rgba(0,0,0,0.35);
}}
.kpi-circle {{
    width:160px; height:160px; border-radius:50%;
    background:conic-gradient(var(--color) calc(var(--value)*1%), #333 0%);
    margin:auto; display:flex; justify-content:center; align-items:center;
    box-shadow:0 0 20px var(--color-glow);
}}
.kpi-circle-inner {{
    width:125px; height:125px; background:#0d0d0d;
    color:white; border-radius:50%; font-size:30px;
    display:flex; justify-content:center; align-items:center;
    font-weight:700;
}}
.metric-bar {{
    background:#2b2b2b; border-radius:12px; height:20px;
}}
.metric-bar-fill {{
    height:100%; border-radius:12px;
    animation:fillBar 1.8s ease forwards;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# HEADER + LOTTIE
# --------------------------------------------------------------
lo = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
colA, colB = st.columns([1,4])

with colA:
    if lo:
        st_lottie(lo, height=140)

with colB:
    st.markdown(f"<h1 style='margin:0;color:{C['text']}'>🚀 Ultra-Premium Animated MT Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{C['muted']}'>3D Animated Metrics • Gradient Lines • Transparent Planes • Mesh Boundaries</p>", unsafe_allow_html=True)

st.write("---")

# --------------------------------------------------------------
# INPUT
# --------------------------------------------------------------
text = st.text_area("Enter text:", height=120)


# --------------------------------------------------------------
# METRIC FUNCTION
# --------------------------------------------------------------
def get_metrics(out, src):
    bleu = compute_bleu(src, out)
    efc = compute_efc(src, out)
    halluc = round(1 - efc, 3)
    semantic = round((bleu + efc) / 2, 3)
    return {"BLEU": bleu, "EFC": efc, "Hallucination": halluc, "Semantic": semantic}


# --------------------------------------------------------------
# RUN BUTTON
# --------------------------------------------------------------
if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Please enter text.")
    else:

        # --------------------------------------------------------------
        # HIDDEN MODEL EXECUTION
        # --------------------------------------------------------------
        outB = baseline_translate(text)
        outE = eact_translate(text)
        outR = rgcld_translate(text)

        mB = get_metrics(outB, text)
        mE = get_metrics(outE, text)
        mR = get_metrics(outR, text)

        # --------------------------------------------------------------
        # TABS LAYOUT
        # --------------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💎 KPI Rings",
            "📈 3D Animated Metrics",
            "🧭 Radar Chart",
            "🧬 Hallucination Metrics",
            "📊 Comparison Table"
        ])

        # **************************************************************
        # TAB 1 — ANIMATED KPI CIRCLES
        # **************************************************************
        with tab1:
            c1, c2, c3 = st.columns(3)
            for col, (name, model, accent) in zip(
                [c1, c2, c3],
                [
                    ("Baseline", mB, C['accent1']),
                    ("EACT", mE, C['accent2']),
                    ("RG-CLD", mR, C['accent3'])
                ],
            ):
                val = model["BLEU"]
                col.markdown(f"""
                <div class="kpi-glass">
                    <h3 style='text-align:center;color:{accent}'>{name}</h3>
                    <div class="kpi-circle" style="--value:{val*100}; --color:{accent}; --color-glow:{accent}55;">
                        <div class="kpi-circle-inner">{val}</div>
                    </div>
                    <p style='text-align:center;color:{C['muted']}'>BLEU Score</p>
                </div>
                """, unsafe_allow_html=True)

        # **************************************************************
        # TAB 2 — 3D ANIMATED METRICS
        # **************************************************************
        with tab2:
            st.markdown("### ✨ 3D Animated Metric Trajectory")
            BLEU = [mB["BLEU"], mE["BLEU"], mR["BLEU"]]
            EFC = [mB["EFC"], mE["EFC"], mR["EFC"]]
            SEM = [mB["Semantic"], mE["Semantic"], mR["Semantic"]]
            labels = ["Baseline", "EACT", "RG-CLD"]

            frames = []
            for i in range(1, 4):
                frames.append(go.Frame(
                    data=[go.Scatter3d(
                        x=BLEU[:i],
                        y=EFC[:i],
                        z=SEM[:i],
                        mode="lines+markers",
                        line=dict(width=6, dash="dash", color=[0, 0.5, 1]),
                        marker=dict(size=8, color=BLEU[:i], colorscale="Electric"),
                        text=labels[:i]
                    )]
                ))

            fig3d = go.Figure(
                data=[go.Scatter3d(
                    x=[BLEU[0]], y=[EFC[0]], z=[SEM[0]],
                    mode="lines+markers",
                    line=dict(width=6, dash="dash"),
                    marker=dict(size=8)
                )],
                frames=frames
            )

            fig3d.update_layout(
                scene=dict(
                    xaxis_title="BLEU",
                    yaxis_title="EFC",
                    zaxis_title="Semantic",
                    xaxis=dict(showbackground=True, backgroundcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(showbackground=True, backgroundcolor="rgba(255,255,255,0.05)"),
                    zaxis=dict(showbackground=True, backgroundcolor="rgba(255,255,255,0.05)"),
                ),
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [{
                        "label": "▶ Play Animation",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 900, "redraw": True}}]
                    }]
                }],
                margin=dict(l=0, r=0, t=0, b=0),
                height=650
            )

            # Mesh surfaces for metric boundaries
            fig3d.add_trace(go.Mesh3d(
                x=[0, 1, 1], y=[0, 0, 1], z=[0, 1, 0],
                opacity=0.1, color=C['accent1']
            ))

            fig3d.add_trace(go.Mesh3d(
                x=[1, 0, 1], y=[1, 1, 0], z=[0, 1, 1],
                opacity=0.1, color=C['accent3']
            ))

            st.plotly_chart(fig3d, use_container_width=True)

        # **************************************************************
        # TAB 3 — RADAR CHART
        # **************************************************************
        with tab3:
            categories = ["BLEU", "EFC", "Hallucination", "Semantic"]

            figRadar = go.Figure()
            figRadar.add_trace(go.Scatterpolar(
                r=[mB[c] for c in categories],
                theta=categories, fill="toself", name="Baseline"
            ))
            figRadar.add_trace(go.Scatterpolar(
                r=[mE[c] for c in categories],
                theta=categories, fill="toself", name="EACT"
            ))
            figRadar.add_trace(go.Scatterpolar(
                r=[mR[c] for c in categories],
                theta=categories, fill="toself", name="RG-CLD"
            ))

            figRadar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=650,
            )

            st.plotly_chart(figRadar, use_container_width=True)

        # **************************************************************
        # TAB 4 — HALLUCINATION TAB (BACK)
        # **************************************************************
        with tab4:
            st.markdown("### 🧬 Hallucination & Semantic Similarity")

            colA, colB = st.columns(2)

            # Hallucination Rate Bars
            with colA:
                st.write("#### Hallucination Rate")
                for name, m, color in [
                    ("Baseline", mB, "#ff4e50"),
                    ("EACT", mE, "#ff9500"),
                    ("RG-CLD", mR, "#ff2a68")
                ]:
                    st.markdown(f"""
                    <p><b>{name}</b>: {m['Hallucination']}</p>
                    <div class="metric-bar">
                        <div class="metric-bar-fill" style="--width:{m['Hallucination']*100}%; background:{color};"></div>
                    </div><br>
                    """, unsafe_allow_html=True)

            # Semantic Similarity Bars
            with colB:
                st.write("#### Semantic Similarity")
                for name, m, color in [
                    ("Baseline", mB, "#4facfe"),
                    ("EACT", mE, "#43e97b"),
                    ("RG-CLD", mR, "#fa709a")
                ]:
                    st.markdown(f"""
                    <p><b>{name}</b>: {m['Semantic']}</p>
                    <div class="metric-bar">
                        <div class="metric-bar-fill" style="--width:{m['Semantic']*100}%; background:{color};"></div>
                    </div><br>
                    """, unsafe_allow_html=True)

        # **************************************************************
        # TAB 5 — COMPARISON TABLE
        # **************************************************************
        with tab5:
            st.write("### 📊 Comparison Table")
            st.table({
                "Model": ["Baseline", "EACT", "RG-CLD"],
                "BLEU": [mB["BLEU"], mE["BLEU"], mR["BLEU"]],
                "EFC": [mB["EFC"], mE["EFC"], mR["EFC"]],
                "Hallucination": [mB["Hallucination"], mE["Hallucination"], mR["Hallucination"]],
                "Semantic": [mB["Semantic"], mE["Semantic"], mR["Semantic"]],
            })
