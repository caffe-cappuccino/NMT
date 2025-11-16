# ===============================================================
#     PREMIUM MT DASHBOARD — Neural Metrics Only
#     Metrics included:
#     - BLEU (surface)
#     - BERTScore F1
#     - Sentence-BERT Similarity (cosine)
#     - Semantic Similarity (BERTScore + SBERT avg)
#     - Hallucination Score (1 – Semantic)
# ===============================================================

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
from sentence_transformers import SentenceTransformer, util

# Lightweight neural models
try:
    from bert_score import score as bert_score_fn
except:
    bert_score_fn = None

import torch

# Your models
from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate

# Simple BLEU
from utils.scoring import compute_bleu

# -----------------------------------------------------------
# Load SBERT model once
# -----------------------------------------------------------
@st.cache_resource
def load_sbert():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sbert_model = load_sbert()

# -----------------------------------------------------------
# Page Config
# -----------------------------------------------------------
st.set_page_config(page_title="Neural MT Dashboard", layout="wide")

# -----------------------------------------------------------
# Lottie Loader
# -----------------------------------------------------------
def load_lottie(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

# -----------------------------------------------------------
# Theme System
# -----------------------------------------------------------
THEMES = {
    "Dark": {
        "bg": "linear-gradient(135deg, #0b1020, #131b2f)",
        "card": "rgba(255,255,255,0.07)",
        "text": "#FFFFFF",
        "muted": "#BFC8D6",
        "acc1": "#4facfe",
        "acc2": "#43e97b",
        "acc3": "#fa709a"
    },
    "Light": {
        "bg": "linear-gradient(135deg, #ffffff, #f0f7ff)",
        "card": "rgba(0,0,0,0.05)",
        "text": "#101624",
        "muted": "#444",
        "acc1": "#0b78d1",
        "acc2": "#16a34a",
        "acc3": "#d63384"
    }
}

theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])
C = THEMES[theme]

# -----------------------------------------------------------
# CSS
# -----------------------------------------------------------
st.markdown(f"""
<style>
body {{ background:{C['bg']}; color:{C['text']}; }}

.kpi-glass {{
  background:{C['card']};
  padding:20px;
  border-radius:14px;
  backdrop-filter:blur(10px);
  border:1px solid rgba(255,255,255,0.18);
}}

.kpi-circle {{
  width:140px;height:140px;border-radius:50%;
  background:conic-gradient(var(--color) calc(var(--value) * 1%), #333 0%);
  display:flex;align-items:center;justify-content:center;margin:auto;
}}

.kpi-circle-inner {{
  width:100px;height:100px;border-radius:50%;
  background:#0d0d0d;color:white;display:flex;
  align-items:center;justify-content:center;font-size:26px;font-weight:700;
}}

.metric-bar {{
  height:16px;border-radius:10px;background:#333;
}}

.metric-bar-fill {{
  height:100%;border-radius:10px;animation:fillBar 1.6s ease forwards;
}}

@keyframes fillBar {{ from {{width:0%;}} to {{width:var(--width);}} }}

.small-muted {{ color:{C['muted']}; font-size:13px; }}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Header with Lottie
# -----------------------------------------------------------
lottie = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")

h1, h2 = st.columns([1,4])
with h1:
    if lottie: st_lottie(lottie, height=120)
with h2:
    st.markdown(f"<h1 style='color:{C['text']}'>🚀 Neural MT Evaluation Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='small-muted'>BERTScore • SBERT Similarity • Semantic Score • Hallucination</p>", unsafe_allow_html=True)

st.write("---")

# -----------------------------------------------------------
# Input Box
# -----------------------------------------------------------
text = st.text_area("Enter text:", height=140)

# -----------------------------------------------------------
# Metric functions
# -----------------------------------------------------------

def bert_score_single(ref, hyp):
    if bert_score_fn is None:
        return None
    P, R, F1 = bert_score_fn([hyp], [ref], lang="en", rescale_with_baseline=True)
    return round(float(F1[0]), 3)

def sbert_similarity(ref, hyp):
    emb_ref = sbert_model.encode(ref, convert_to_tensor=True)
    emb_hyp = sbert_model.encode(hyp, convert_to_tensor=True)
    sim = util.cos_sim(emb_ref, emb_hyp).item()
    return round(float(sim), 3)

def compute_all(ref, hyp):
    bleu = compute_bleu(ref, hyp)  # small, surface metric
    bert = bert_score_single(ref, hyp)
    sbert = sbert_similarity(ref, hyp)

    # Semantic score = average of two neural similarities
    semantic = round((bert + sbert) / 2, 3)

    halluc = round(1 - semantic, 3)

    return {
        "BLEU": bleu,
        "BERT": bert,
        "SBERT": sbert,
        "Semantic": semantic,
        "Hallucination": halluc
    }

# -----------------------------------------------------------
# RUN BUTTON
# -----------------------------------------------------------
if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Enter text first.")
    else:
        # run the models
        out_b = baseline_translate(text)
        out_e = eact_translate(text)
        out_r = rgcld_translate(text)

        mB = compute_all(text, out_b)
        mE = compute_all(text, out_e)
        mR = compute_all(text, out_r)

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💎 KPI Rings",
            "📈 3D Trajectory",
            "🧭 Radar",
            "📉 Bars",
            "📊 Table"
        ])

        # -----------------------------------------------------------
        # TAB 1 — KPI RINGS
        # -----------------------------------------------------------
        with tab1:
            c1, c2, c3 = st.columns(3)
            for (name, m, acc), col in zip(
                [("Baseline", mB, C['acc1']),
                 ("EACT", mE, C['acc2']),
                 ("RG-CLD", mR, C['acc3'])],
                [c1, c2, c3]
            ):
                col.markdown(f"""
                <div class='kpi-glass'>
                  <h3 style='color:{acc};text-align:center'>{name}</h3>
                  <div class='kpi-circle' style="--value:{m['Semantic']*100};--color:{acc}">
                    <div class='kpi-circle-inner'>{m['Semantic']}</div>
                  </div>
                  <p class='small-muted' style='text-align:center'>Semantic Score</p>
                  <p class='small-muted'>BERTScore: <b style='color:{C['text']}'>{m['BERT']}</b></p>
                  <p class='small-muted'>SBERT Sim: <b style='color:{C['text']}'>{m['SBERT']}</b></p>
                </div>
                """, unsafe_allow_html=True)

        # -----------------------------------------------------------
        # TAB 2 — 3D TRAJECTORY (Semantic space)
        # -----------------------------------------------------------
        with tab2:
            x = [mB["Semantic"], mE["Semantic"], mR["Semantic"]]
            y = [mB["BERT"], mE["BERT"], mR["BERT"]]
            z = [mB["SBERT"], mE["SBERT"], mR["SBERT"]]
            labels = ["Baseline", "EACT", "RG-CLD"]

            xs = np.linspace(x[0], x[-1], 80)
            ys = np.linspace(y[0], y[-1], 80)
            zs = np.linspace(z[0], z[-1], 80)

            fig = go.Figure()

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(width=6, color=np.linspace(0,1,80), colorscale="Rainbow", dash="dot")
            ))

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=7, color=[0,0.5,1], colorscale="Rainbow")
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title="Semantic",
                    yaxis_title="BERTScore",
                    zaxis_title="SBERT Similarity"
                ),
                height=620
            )
            st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------------------
        # TAB 3 — RADAR
        # -----------------------------------------------------------
        with tab3:
            keys = ["BLEU", "BERT", "SBERT", "Semantic", "Hallucination"]

            def vec(m):
                return [m[k] if m[k] is not None else 0 for k in keys]

            figR = go.Figure()
            figR.add_trace(go.Scatterpolar(r=vec(mB), theta=keys, fill='toself', name='Baseline'))
            figR.add_trace(go.Scatterpolar(r=vec(mE), theta=keys, fill='toself', name='EACT'))
            figR.add_trace(go.Scatterpolar(r=vec(mR), theta=keys, fill='toself', name='RG-CLD'))

            figR.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=620)
            st.plotly_chart(figR, use_container_width=True)

        # -----------------------------------------------------------
        # TAB 4 — Animated Bars
        # -----------------------------------------------------------
        with tab4:
            colA, colB = st.columns(2)

            with colA:
                st.markdown("### Neural Similarities (BERTScore / SBERT)")
                for name, m, clr in [("Baseline", mB, C['acc1']), ("EACT", mE, C['acc2']), ("RG-CLD", mR, C['acc3'])]:
                    v1 = m["BERT"]
                    v2 = m["SBERT"]
                    st.markdown(f"<b>{name}</b>", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class='metric-bar'>
                      <div class='metric-bar-fill' style="--width:{v1*100}%; background:{clr};"></div>
                    </div>
                    <div class='small-muted'>BERTScore: {v1}</div><br>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class='metric-bar'>
                      <div class='metric-bar-fill' style="--width:{v2*100}%; background:{clr};"></div>
                    </div>
                    <div class='small-muted'>SBERT Similarity: {v2}</div><br>
                    """, unsafe_allow_html=True)

            with colB:
                st.markdown("### Semantic + Hallucination")
                for name, m, clr in [("Baseline", mB, "#ff4e50"), ("EACT", mE, "#ff9500"), ("RG-CLD", mR, "#ff2a68")]:
                    sem = m["Semantic"]
                    hal = m["Hallucination"]

                    st.markdown(f"<b>{name}</b>", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class='metric-bar'>
                      <div class='metric-bar-fill' style="--width:{sem*100}%; background:{clr};"></div>
                    </div>
                    <div class='small-muted'>Semantic: {sem}</div><br>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class='metric-bar'>
                      <div class='metric-bar-fill' style="--width:{hal*100}%; background:{clr};"></div>
                    </div>
                    <div class='small-muted'>Hallucination: {hal}</div><br>
                    """, unsafe_allow_html=True)

        # -----------------------------------------------------------
        # TAB 5 — TABLE
        # -----------------------------------------------------------
        with tab5:
            st.write("### Full Metric Table")
            st.table({
                "Model": ["Baseline", "EACT", "RG-CLD"],
                "BLEU": [mB["BLEU"], mE["BLEU"], mR["BLEU"]],
                "BERTScore": [mB["BERT"], mE["BERT"], mR["BERT"]],
                "SBERT Similarity": [mB["SBERT"], mE["SBERT"], mR["SBERT"]],
                "Semantic Score": [mB["Semantic"], mE["Semantic"], mR["Semantic"]],
                "Hallucination": [mB["Hallucination"], mE["Hallucination"], mR["Hallucination"]]
            })

        st.success("Evaluation complete ✔")
