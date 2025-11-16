# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
import time

# Lightweight imports for metrics; some are imported lazily in functions
try:
    from bert_score import score as bert_score_fn
except Exception:
    bert_score_fn = None

try:
    import sacrebleu
    from sacrebleu.metrics import TER as SacreTER
except Exception:
    sacrebleu = None
    SacreTER = None

# COMET imports will be lazy-loaded (lightweight model chosen)
try:
    from comet import download_model, load_from_checkpoint
except Exception:
    download_model = None
    load_from_checkpoint = None

# Your model wrappers (these should exist in models/)
from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate

# Existing BLEU/EFC stubs (utils/scoring.py)
from utils.scoring import compute_bleu, compute_efc

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="MT Dashboard — BLEU EFC BERT COMET TER", layout="wide")

# ---------------------------
# Lottie helper
# ---------------------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

# ---------------------------
# Theme & CSS (keeps premium look)
# ---------------------------
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

st.markdown(f"""
<style>
body {{ background:{C['bg']}; color:{C['text']}; }}
.kpi-glass {{
  background:{C['card_bg']};
  backdrop-filter: blur(10px);
  padding:20px;
  border-radius:12px;
  border:1px solid rgba(255,255,255,0.12);
  box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}}
.kpi-circle {{
    width:130px;height:130px;border-radius:50%;
    background: conic-gradient(var(--color) calc(var(--value) * 1%), #333 0%);
    display:flex;align-items:center;justify-content:center;margin:auto;
    box-shadow:0 0 18px var(--color-glow);
}}
.kpi-circle-inner {{ width:95px;height:95px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;background:#0d0d0d;}}
.metric-bar {{ height:16px;border-radius:10px;background:#333;overflow:hidden; }}
.metric-bar-fill {{ height:100%; border-radius:10px; animation:fillBar 1.6s ease forwards; }}
@keyframes fillBar {{ from {{ width:0%; }} to {{ width:var(--width); }} }}
.small-muted {{ color: {C['muted']}; font-size:13px; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header with lottie
# ---------------------------
lottie_json = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
h1, h2 = st.columns([1,4])
with h1:
    if lottie_json:
        st_lottie(lottie_json, height=120)
with h2:
    st.markdown(f"<h1 style='color:{C['text']}; margin:0;'>🚀 MT Eval — BLEU • EFC • BERTScore • COMET • TER</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>Single-button evaluation · premium visuals · lightweight COMET</div>", unsafe_allow_html=True)
st.write("---")

# ---------------------------
# Input
# ---------------------------
text = st.text_area("Enter text to evaluate (models run in background, translations hidden):", height=140)

# ---------------------------
# Metric helper functions
# ---------------------------
def compute_bertscore_single(ref, hyp, lang="en"):
    """Compute BERTScore F1 for a single pair. Returns float in [0,1]."""
    if bert_score_fn is None:
        return None
    try:
        P, R, F1 = bert_score_fn([hyp], [ref], lang=lang, rescale_with_baseline=True)
        return float(F1[0].cpu().numpy()) if hasattr(F1[0], "cpu") else float(F1[0])
    except Exception as e:
        st.warning(f"BERTScore compute failed: {e}")
        return None

def compute_ter_single(ref, hyp):
    """Compute TER and convert to metric where higher is better: ter_metric = clamp(1 - TER, 0, 1)."""
    if SacreTER is None:
        return None
    try:
        # sacrebleu TER expects corpus format: references is list of reference lists
        ter_metric = SacreTER().corpus_score([hyp], [[ref]]).score  # this returns TER in percentage? sacrebleu returns value between 0..100 sometimes
        # sacrebleu's TER().corpus_score returns a score object; .score gives float maybe percentage
        # To be safe, if ter_metric > 1, convert percentage to 0..1
        if ter_metric > 1.0:
            ter = ter_metric / 100.0
        else:
            ter = ter_metric
        val = max(0.0, min(1.0, 1.0 - ter))
        return round(val, 3)
    except Exception as e:
        st.warning(f"TER compute failed: {e}")
        return None

# COMET: lazy loader + predictor
_COMET_MODEL = None
def load_comet_model_light():
    """Loads a lightweight COMET model. Returns model or None on failure."""
    global _COMET_MODEL
    if _COMET_MODEL is not None:
        return _COMET_MODEL
    if load_from_checkpoint is None:
        return None
    try:
        # Lightweight model (as requested)
        # model_name = "Unbabel/wmt22-cometkiwi-da"  # lightweight-ish
        model_name = "Unbabel/wmt22-cometkiwi-da"
        # download_model can be used if needed; load_from_checkpoint will fetch from HF hub if allowed
        _COMET_MODEL = load_from_checkpoint(model_name)
        return _COMET_MODEL
    except Exception as e:
        st.warning(f"COMET load failed: {e}")
        _COMET_MODEL = None
        return None

def compute_comet_single(src, hyp):
    """Compute COMET score (scalar). Returns float approximately in [0,1] or None."""
    model = load_comet_model_light()
    if model is None:
        return None
    try:
        # COMET model's predict requires a dict depending on model; we try the common API
        preds = model.predict([{"src": src, "mt": hyp, "ref": ""}])
        # The exact return format depends on the COMET version; try to extract a score
        if isinstance(preds, list) and len(preds) > 0:
            # preds might be dict with 'scores' or float
            v = None
            entry = preds[0]
            if isinstance(entry, dict):
                v = entry.get("score") or entry.get("scores") or entry.get("comet_score") or entry.get("preds")
                if isinstance(v, (list, tuple)):
                    v = v[0]
            else:
                v = entry
            if v is None:
                return None
            # normalize broad COMET values to 0..1 via a logistic-ish mapping (if needed)
            try:
                fv = float(v)
            except:
                return None
            # Many COMET models output values roughly in [-1,1] or [-100,100]. We'll map [-1,1] -> [0,1]
            if -1.5 < fv < 1.5:
                mapped = (fv + 1.0) / 2.0
            else:
                # if in wider range, min-max clamp using arctan scaling
                mapped = (np.arctan(fv) / (np.pi/2) + 1) / 2
            return round(float(max(0.0, min(1.0, mapped))), 3)
        else:
            return None
    except Exception as e:
        st.warning(f"COMET predict failed: {e}")
        return None

# ---------------------------
# Combined metric computation
# ---------------------------
def compute_all_metrics(ref, hyp, bert_lang="en"):
    """
    returns dict with:
      BLEU, EFC, BERT, COMET, TER, Hallucination (1 - EFC), Semantic (avg BLEU+EFC)
    Any missing metric will be None.
    """
    out = {}
    try:
        out['BLEU'] = compute_bleu(ref, hyp)
    except Exception:
        out['BLEU'] = None
    try:
        out['EFC'] = compute_efc(ref, hyp)
    except Exception:
        out['EFC'] = None

    # Hallucination
    if out.get('EFC') is not None:
        out['Hallucination'] = round(max(0.0, min(1.0, 1.0 - out['EFC'])), 3)
    else:
        out['Hallucination'] = None

    # Semantic = avg BLEU + EFC
    if out.get('BLEU') is not None and out.get('EFC') is not None:
        out['Semantic'] = round((out['BLEU'] + out['EFC']) / 2.0, 3)
    else:
        out['Semantic'] = None

    # BERTScore
    bert = compute_bertscore_single(ref, hyp) if bert_score_fn is not None else None
    out['BERT'] = round(bert, 3) if bert is not None else None

    # TER
    ter_m = compute_ter_single(ref, hyp) if SacreTER is not None else None
    out['TER'] = ter_m if ter_m is not None else None

    # COMET
    comet_m = compute_comet_single(ref, hyp)  # may return None if COMET unavailable
    out['COMET'] = comet_m if comet_m is not None else None

    return out

# ---------------------------
# Main UI: single Run Evaluation button + Tabs (premium)
# ---------------------------
if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Please enter text to evaluate.")
    else:
        # run models (hidden)
        b_out = baseline_translate(text)
        e_out = eact_translate(text)
        r_out = rgcld_translate(text)

        m_b = compute_all_metrics(text, b_out)
        m_e = compute_all_metrics(text, e_out)
        m_r = compute_all_metrics(text, r_out)

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💎 KPI Rings",
            "📈 3D Trajectory",
            "🧭 Radar",
            "📉 Advanced Bars",
            "📊 Table"
        ])

        # Tab1: KPI Rings (include BERT & COMET small values under)
        with tab1:
            col1, col2, col3 = st.columns(3)
            for (name, m, acc), col in zip(
                [("Baseline", m_b, C['acc1']),
                 ("EACT", m_e, C['acc2']),
                 ("RG-CLD", m_r, C['acc3'])],
                [col1, col2, col3]
            ):
                bleu_val = m.get('BLEU') or 0.0
                bert_val = m.get('BERT')
                comet_val = m.get('COMET')
                col.markdown(f"""
                    <div class='kpi-glass'>
                        <h3 style='color:{acc}; margin:0;'>{name}</h3>
                        <div class="kpi-circle" style="--value:{bleu_val*100}; --color:{acc}; --color-glow:{acc}66;">
                          <div class="kpi-circle-inner">{bleu_val}</div>
                        </div>
                        <p style='text-align:center; color:{C["muted"]}; margin:6px 0 0 0;'>BLEU Score</p>
                        <div style='margin-top:8px; display:flex; justify-content:space-between;'>
                            <div style='color:{C["muted"]};'>BERT: <b style='color:{C["text"]};'>{bert_val if bert_val is not None else "n/a"}</b></div>
                            <div style='color:{C["muted"]};'>COMET: <b style='color:{C["text"]};'>{comet_val if comet_val is not None else "n/a"}</b></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Tab2: 3D Trajectory (lines) — include color gradient and dotted style (as earlier)
        with tab2:
            st.markdown("### 3D Metric Trajectory — BLEU → EFC → Semantic")
            x = [m_b['BLEU'] or 0.0, m_e['BLEU'] or 0.0, m_r['BLEU'] or 0.0]
            y = [m_b['EFC'] or 0.0, m_e['EFC'] or 0.0, m_r['EFC'] or 0.0]
            z = [m_b['Semantic'] or 0.0, m_e['Semantic'] or 0.0, m_r['Semantic'] or 0.0]
            labels = ["Baseline", "EACT", "RG-CLD"]

            # smooth interpolated points
            steps = 60
            xs = np.linspace(x[0], x[-1], steps)
            ys = np.linspace(y[0], y[-1], steps)
            zs = np.linspace(z[0], z[-1], steps)

            fig = go.Figure()

            # gradient colored line: use color array
            color_vals = np.linspace(0, 1, steps)
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(width=6, color=color_vals, colorscale='Jet', dash='dot'),
                name='Trajectory'
            ))

            # markers for model points
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode='markers+text',
                marker=dict(size=6, color=[0,0.5,1], colorscale='Jet'),
                text=labels, textposition='top center', name='Models'
            ))

            # transparent planes (XY, YZ, XZ)
            plane_size = np.linspace(0,1,4)
            xx, yy = np.meshgrid(plane_size, plane_size)
            # XY plane at z=min(z)-0.02
            zz = np.ones_like(xx) * (min(z)-0.02 if min(z)>0.02 else 0.0)
            fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.12, colorscale='Greys', showscale=False, name='XZ Plane'))

            # YZ plane (x constant)
            yy2, zz2 = np.meshgrid(plane_size, plane_size)
            xx2 = np.ones_like(yy2) * (min(x)-0.02 if min(x)>0.02 else 0.0)
            fig.add_trace(go.Surface(x=xx2, y=yy2, z=zz2, opacity=0.08, colorscale='Blues', showscale=False, name='YZ Plane'))

            fig.update_layout(scene=dict(
                xaxis_title='BLEU',
                yaxis_title='EFC',
                zaxis_title='Semantic'),
                height=650, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Tab3: Radar — expand axes to include BERT and COMET and TER (TER normalized)
        with tab3:
            keys = ["BLEU", "EFC", "BERT", "COMET", "TER", "Hallucination", "Semantic"]
            def get_vals(m):
                vals = []
                for k in keys:
                    v = m.get(k)
                    if v is None:
                        vals.append(0.0)
                    else:
                        vals.append(float(v))
                return vals

            figR = go.Figure()
            figR.add_trace(go.Scatterpolar(r=get_vals(m_b), theta=keys, fill='toself', name='Baseline'))
            figR.add_trace(go.Scatterpolar(r=get_vals(m_e), theta=keys, fill='toself', name='EACT'))
            figR.add_trace(go.Scatterpolar(r=get_vals(m_r), theta=keys, fill='toself', name='RG-CLD'))
            figR.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=650)
            st.plotly_chart(figR, use_container_width=True)

        # Tab4: Advanced bars — show BERT, COMET, TER too
        with tab4:
            cA, cB = st.columns(2)
            with cA:
                st.markdown("### Neural Metrics: BERTScore & COMET")
                for name, m, color in [("Baseline", m_b, C['acc1']), ("EACT", m_e, C['acc2']), ("RG-CLD", m_r, C['acc3'])]:
                    bertv = m.get('BERT') if m.get('BERT') is not None else 0.0
                    cometv = m.get('COMET') if m.get('COMET') is not None else 0.0
                    st.markdown(f"<b>{name}</b>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="metric-bar">
                          <div class="metric-bar-fill" style="--width:{bertv*100}%; background:{color};"></div>
                        </div>
                        <div class='small-muted'>BERTScore: {bertv if m.get('BERT') is not None else 'n/a'}</div>
                        <div style='height:10px'></div>
                        <div class="metric-bar">
                          <div class="metric-bar-fill" style="--width:{cometv*100}%; background:{C['muted']};"></div>
                        </div>
                        <div class='small-muted'>COMET: {cometv if m.get('COMET') is not None else 'n/a'}</div>
                        <br>
                    """, unsafe_allow_html=True)

            with cB:
                st.markdown("### TER (converted → higher is better)")
                for name, m, color in [("Baseline", m_b, "#ff4e50"), ("EACT", m_e, "#ffa600"), ("RG-CLD", m_r, "#ff2a68")]:
                    terv = m.get('TER') if m.get('TER') is not None else 0.0
                    st.markdown(f"<b>{name}</b>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="metric-bar">
                          <div class="metric-bar-fill" style="--width:{terv*100}%; background:{color};"></div>
                        </div>
                        <div class='small-muted'>TER_converted: {terv if m.get('TER') is not None else 'n/a'}</div><br>
                    """, unsafe_allow_html=True)

        # Tab5: Table
        with tab5:
            st.write("### Complete metric table")
            table = {
                "Model": ["Baseline", "EACT", "RG-CLD"],
                "BLEU": [m_b.get('BLEU'), m_e.get('BLEU'), m_r.get('BLEU')],
                "EFC": [m_b.get('EFC'), m_e.get('EFC'), m_r.get('EFC')],
                "BERTScore": [m_b.get('BERT'), m_e.get('BERT'), m_r.get('BERT')],
                "COMET": [m_b.get('COMET'), m_e.get('COMET'), m_r.get('COMET')],
                "TER_converted (1-TER)": [m_b.get('TER'), m_e.get('TER'), m_r.get('TER')],
                "Hallucination": [m_b.get('Hallucination'), m_e.get('Hallucination'), m_r.get('Hallucination')],
                "Semantic": [m_b.get('Semantic'), m_e.get('Semantic'), m_r.get('Semantic')]
            }
            st.table(table)

        st.success("Evaluation complete ✅")
