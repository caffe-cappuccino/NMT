import streamlit as st
import plotly.graph_objects as go
import numpy as np
import requests
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt

from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate
from utils.scoring import compute_bleu, compute_efc

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Next-Level MT Dashboard", layout="wide")

# ---------------------------
# Load Lottie
# ---------------------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

# ---------------------------
# Themes
# ---------------------------
THEMES = {
    "Dark": {
        "bg": "linear-gradient(135deg, #0b1020, #0f1724)",
        "card": "rgba(255,255,255,0.06)",
        "text": "#FFFFFF",
        "muted": "#BFC8D6",
        "a1": "#4facfe",
        "a2": "#43e97b",
        "a3": "#fa709a"
    },
    "Light": {
        "bg": "linear-gradient(135deg, #ffffff, #f0f6ff)",
        "card": "rgba(0,0,0,0.04)",
        "text": "#111111",
        "muted": "#444444",
        "a1": "#0b78d1",
        "a2": "#16a34a",
        "a3": "#d63384"
    }
}

chosen_theme = st.sidebar.selectbox("💡 AI Theme Mode", ["Dark", "Light"])
T = THEMES[chosen_theme]

# ---------------------------
# Inject Theme CSS
# ---------------------------
st.markdown(f"""
<style>

body {{
    background: {T['bg']};
    color: {T['text']};
}}

.kpi {{
    background: {T['card']};
    padding: 22px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 5px 25px rgba(0,0,0,0.25);
}}

.small {{
    color: {T['muted']};
    font-size: 13px;
}}

.metric-circle {{
    width: 150px;
    height: 150px;
    border-radius: 50%;
    background: conic-gradient(var(--color) calc(var(--value) * 1%), #333 0%);
    display: flex;
    justify-content: center;
    align-items: center;
    margin: auto;
    box-shadow: 0 0 20px var(--color);
}}

.circle-inner {{
    width: 110px;
    height: 110px;
    background: #0d0d0d;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 24px;
}}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header Section
# ---------------------------
colA, colB = st.columns([1,4])
with colA:
    anim = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
    if anim:
        st_lottie(anim, height=120)

with colB:
    st.markdown(f"""
        <h1 style='margin:0; color:{T['text']}'>
            🚀 Next-Level Animated MT Evaluation Dashboard
        </h1>
        <div class='small'>AI Theme · 3D Charts · Radar · Lottie Animations</div>
    """, unsafe_allow_html=True)

st.write("---")

# ---------------------------
# User Input
# ---------------------------
text = st.text_area("Enter text to evaluate:", height=140)

# ===========================
# SINGLE BUTTON (Run Evaluation)
# ===========================
if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Please enter some text.")
    else:
        # ---------------------------
        # MODEL RUNS
        # ---------------------------
        b_out = baseline_translate(text)
        e_out = eact_translate(text)
        r_out = rgcld_translate(text)

        # ---------------------------
        # FIND METRICS
        # ---------------------------
        def metrics(output):
            bleu = compute_bleu(text, output)
            efc = compute_efc(text, output)
            hall = round(1 - efc, 3)
            sim = round((bleu + efc) / 2, 3)
            return bleu, efc, hall, sim

        b_bleu, b_efc, b_hall, b_sim = metrics(b_out)
        e_bleu, e_efc, e_hall, e_sim = metrics(e_out)
        r_bleu, r_efc, r_hall, r_sim = metrics(r_out)

        # ---------------------------
        # KPI CARDS (Animated Circles)
        # ---------------------------
        st.subheader("💎 Overview Metrics")

        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown(f"""
            <div class='kpi'>
                <h3 style='color:{T['a1']}'>Baseline</h3>
                <div class='metric-circle' style="--value:{b_bleu*100}; --color:{T['a1']}">
                    <div class='circle-inner'>{b_bleu}</div>
                </div>
                <p class='small'>BLEU Score</p>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class='kpi'>
                <h3 style='color:{T['a2']}'>EACT</h3>
                <div class='metric-circle' style="--value:{e_bleu*100}; --color:{T['a2']}">
                    <div class='circle-inner'>{e_bleu}</div>
                </div>
                <p class='small'>BLEU Score</p>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class='kpi'>
                <h3 style='color:{T['a3']}'>RG-CLD</h3>
                <div class='metric-circle' style="--value:{r_bleu*100}; --color:{T['a3']}">
                    <div class='circle-inner'>{r_bleu}</div>
                </div>
                <p class='small'>BLEU Score</p>
            </div>
            """, unsafe_allow_html=True)

        # ---------------------------
        # TABS for Different Metric Sections
        # ---------------------------
        tab1, tab2, tab3 = st.tabs(["📈 Primary Metrics", "🧠 Advanced Metrics", "📊 Comparison Table"])

        # ---------------------------
        # TAB 1 — PRIMARY METRICS
        # ---------------------------
        with tab1:
            st.write("### BLEU & EFC Comparison")

            c1, c2, c3 = st.columns(3)

            # Base
            with c1:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["BLEU", "EFC"], [b_bleu, b_efc], color=[T['a1'], T['a1']])
                ax.set_ylim(0,1)
                ax.set_title("Baseline")
                st.pyplot(fig)

            # EACT
            with c2:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["BLEU", "EFC"], [e_bleu, e_efc], color=[T['a2'], T['a2']])
                ax.set_ylim(0,1)
                ax.set_title("EACT")
                st.pyplot(fig)

            # RG-CLD
            with c3:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["BLEU", "EFC"], [r_bleu, r_efc], color=[T['a3'], T['a3']])
                ax.set_ylim(0,1)
                ax.set_title("RG-CLD")
                st.pyplot(fig)

        # ---------------------------
        # TAB 2 — ADVANCED METRICS
        # ---------------------------
        with tab2:
            st.write("### Hallucination & Semantic Similarity")

            a1, a2 = st.columns(2)

            with a1:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["Baseline","EACT","RG-CLD"], [b_hall,e_hall,r_hall], color=T['a3'])
                ax.set_ylim(0,1)
                ax.set_title("Hallucination Rate")
                st.pyplot(fig)

            with a2:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["Baseline","EACT","RG-CLD"], [b_sim,e_sim,r_sim], color=T['a1'])
                ax.set_ylim(0,1)
                ax.set_title("Semantic Similarity")
                st.pyplot(fig)

            # Radar Chart
            st.write("### Radar Comparison Chart")
            categories = ["BLEU","EFC","Hallucination","Semantic"]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=[b_bleu,b_efc,b_hall,b_sim], theta=categories, fill='toself', name="Baseline"))
            fig_radar.add_trace(go.Scatterpolar(r=[e_bleu,e_efc,e_hall,e_sim], theta=categories, fill='toself', name="EACT"))
            fig_radar.add_trace(go.Scatterpolar(r=[r_bleu,r_efc,r_hall,r_sim], theta=categories, fill='toself', name="RG-CLD"))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])))
            st.plotly_chart(fig_radar, use_container_width=True)

        # ---------------------------
        # TAB 3 — COMPARISON TABLE
        # ---------------------------
        with tab3:
            st.table({
                "Model":["Baseline","EACT","RG-CLD"],
                "BLEU":[b_bleu,e_bleu,r_bleu],
                "EFC":[b_efc,e_efc,r_efc],
                "Hallucination":[b_hall,e_hall,r_hall],
                "Semantic":[b_sim,e_sim,r_sim]
            })
