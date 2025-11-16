# app.py — COMET REMOVED + BERTScore + TER + BLEU + EFC

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

# Optional metrics
try:
    from bert_score import score as bert_score_fn
except:
    bert_score_fn = None

try:
    import sacrebleu
    from sacrebleu.metrics import TER as SacreTER
except:
    sacrebleu = None
    SacreTER = None

# Your models
from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate

# Your metrics
from utils.scoring import compute_bleu, compute_efc

# -----------------------------------------------------------
# Page Config
# -----------------------------------------------------------
st.set_page_config(page_title="MT Dashboard (No COMET)", layout="wide")

# -----------------------------------------------------------
# Lottie Loader
# -----------------------------------------------------------
def load_lottie(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

# -----------------------------------------------------------
# Themes
# -----------------------------------------------------------
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
body {{
  background: {C['bg']};
  color: {C['text']};
}}
.kpi-glass {{
  background: {C['card_bg']};
  padding: 20px;
  border-radius: 14px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.18);
}}
.kpi-circle {{
  width: 140px; height: 140px;
  border-radius: 50%;
  background: conic-gradient(var(--color) calc(var(--value) * 1%), #333 0%);
  display:flex; align-items:center; justify-content:center;
  margin:auto;
}}
.kpi-circle-inner {{
  width:100px;height:100px;
  border-radius:50%;
  background:#0d0d0d;
  color:white;
  display:flex;align-items:center;justify-content:center;
  font-size:26px;font-weight:700;
}}
.metric-bar {{
  height:16px;border-radius:10px;background:#333;
}}
.metric-bar-fill {{
  height:100%;border-radius:10px;
  animation:fillBar 1.7s ease forwards;
}}
@keyframes fillBar {{
  from {{width:0%;}}
  to {{width:var(--width);}}
}}
.small-muted {{ color:{C['muted']}; font-size:13px; }}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Header
# -----------------------------------------------------------
lottie = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
a, b = st.columns([1,4])
with a:
    if lottie:
        st_lottie(lottie, height=120)
with b:
    st.markdown(f"<h1 style='color:{C['text']}'>🚀 MT Dashboard (BERTScore + TER)</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='small-muted'>COMET removed · Premium UI stays</p>", unsafe_allow_html=True)

st.write("---")

# -----------------------------------------------------------
# Input
# -----------------------------------------------------------
text = st.text_area("Enter text:", height=140)

# -----------------------------------------------------------
# Metric Functions
# -----------------------------------------------------------
def compute_bert(ref, hyp):
    if bert_score_fn is None:
        return None
    try:
        P, R, F1 = bert_score_fn([hyp], [ref], lang="en", rescale_with_baseline=True)
        return round(float(F1[0]), 3)
    except:
        return None

def compute_ter(ref, hyp):
    if SacreTER is None:
        return None
    try:
        score = SacreTER().corpus_score([hyp], [[ref]]).score
        if score > 1:
            score = score / 100.0
        return round(max(0, min(1, 1 - score)), 3)
    except:
        return None

def compute_all(ref, hyp):
    bleu = compute_bleu(ref, hyp)
    efc = compute_efc(ref, hyp)
    halluc = round(1 - efc, 3)
    semantic = round((bleu + efc) / 2, 3)
    bert = compute_bert(ref, hyp)
    ter = compute_ter(ref, hyp)
    return {
        "BLEU": bleu,
        "EFC": efc,
        "Hallucination": halluc,
        "Semantic": semantic,
        "BERT": bert,
        "TER": ter
    }

# -----------------------------------------------------------
# Run Evaluation
# -----------------------------------------------------------
if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Enter text first.")
    else:
        out_b = baseline_translate(text)
        out_e = eact_translate(text)
        out_r = rgcld_translate(text)

        mB = compute_all(text, out_b)
        mE = compute_all(text, out_e)
        mR = compute_all(text, out_r)

        # TABS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💎 KPI Rings", "📈 3D Trajectory",
            "🧭 Radar", "📉 Bars", "📊 Table"
        ])

        # ---------------- TAB 1: KPI Rings ----------------
        with tab1:
            c1, c2, c3 = st.columns(3)
            for (name, m, acc), col in zip(
                [("Baseline", mB, C['acc1']),
                 ("EACT", mE, C['acc2']),
                 ("RG-CLD", mR, C['acc3'])],
                [c1, c2, c3]
            ):
                bleu = m["BLEU"]
                bert = m["BERT"]
                ter = m["TER"]
                col.markdown(f"""
                <div class='kpi-glass'>
                    <h3 style='color:{acc};text-align:center'>{name}</h3>
                    <div class='kpi-circle' style="--value:{bleu*100};--color:{acc};--color-glow:{acc}">
                        <div class='kpi-circle-inner'>{bleu}</div>
                    </div>
                    <p class='small-muted' style='text-align:center'>BLEU</p>
                    <p class='small-muted'>BERTScore: <b style='color:{C['text']}'>{bert}</b></p>
                    <p class='small-muted'>TER (1-TER): <b style='color:{C['text']}'>{ter}</b></p>
                </div>
                """, unsafe_allow_html=True)

        # ---------------- TAB 2: 3D Trajectory ----------------
        with tab2:
            x = [mB["BLEU"], mE["BLEU"], mR["BLEU"]]
            y = [mB["EFC"], mE["EFC"], mR["EFC"]]
            z = [mB["Semantic"], mE["Semantic"], mR["Semantic"]]
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
                    xaxis_title="BLEU",
                    yaxis_title="EFC",
                    zaxis_title="Semantic"
                ),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---------------- TAB 3: Radar ----------------
        with tab3:
            keys = ["BLEU", "EFC", "BERT", "TER", "Hallucination", "Semantic"]

            def vec(m):
                return [m[k] if m[k] is not None else 0 for k in keys]

            figR = go.Figure()
            figR.add_trace(go.Scatterpolar(r=vec(mB), theta=keys, fill='toself', name='Baseline'))
            figR.add_trace(go.Scatterpolar(r=vec(mE), theta=keys, fill='toself', name='EACT'))
            figR.add_trace(go.Scatterpolar(r=vec(mR), theta=keys, fill='toself', name='RG-CLD'))

            figR.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=600)
            st.plotly_chart(figR, use_container_width=True)

        # ---------------- TAB 4: Bars ----------------
        with tab4:
            colA, colB = st.columns(2)

            with colA:
                st.markdown("### BERTScore")
                for name, m, color in [("Baseline", mB, C['acc1']), ("EACT", mE, C['acc2']), ("RG-CLD", mR, C['acc3'])]:
                    v = m["BERT"] or 0
                    st.markdown(f"<b>{name}</b>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='metric-bar'>
                        <div class='metric-bar-fill' style="--width:{v*100}%; background:{color};"></div>
                    </div>
                    <div class='small-muted'>BERT: {v}</div><br>
                    """, unsafe_allow_html=True)

            with colB:
                st.markdown("### TER (converted)")
                for name, m, color in [("Baseline", mB, "#ff4e50"), ("EACT", mE, "#ffa600"), ("RG-CLD", mR, "#ff2a68")]:
                    v = m["TER"] or 0
                    st.markdown(f"<b>{name}</b>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='metric-bar'>
                        <div class='metric-bar-fill' style="--width:{v*100}%; background:{color};"></div>
                    </div>
                    <div class='small-muted'>TER_converted: {v}</div><br>
                    """, unsafe_allow_html=True)

        # ---------------- TAB 5: Table ----------------
        with tab5:
            st.write("### Complete Metrics Table")
            st.table({
                "Model": ["Baseline", "EACT", "RG-CLD"],
                "BLEU": [mB["BLEU"], mE["BLEU"], mR["BLEU"]],
                "EFC": [mB["EFC"], mE["EFC"], mR["EFC"]],
                "BERT": [mB["BERT"], mE["BERT"], mR["BERT"]],
                "TER (1-TER)": [mB["TER"], mE["TER"], mR["TER"]],
                "Hallucination": [mB["Hallucination"], mE["Hallucination"], mR["Hallucination"]],
                "Semantic": [mB["Semantic"], mE["Semantic"], mR["Semantic"]],
            })

        st.success("Evaluation complete ✔")
