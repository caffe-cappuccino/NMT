# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
import time
import json
import streamlit.components.v1 as components

# Import your model wrappers and scoring utils
from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate
from utils.scoring import compute_bleu, compute_efc

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Neural Translation Evaluation & Insights Platform", layout="wide")

# ---------------------------------------------------------------------
# Helpers: load Lottie
# ---------------------------------------------------------------------
def load_lottie(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

# loading animation (used while processing)
loading_animation = load_lottie("https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json")
header_lottie = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")

# ---------------------------------------------------------------------
# Themes (colors)
# ---------------------------------------------------------------------
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
        "bg": "linear-gradient(135deg, #ffffff, #f0f7ff)",
        "card_bg": "rgba(0,0,0,0.05)",
        "text": "#101624",
        "muted": "#444444",
        "accent1": "#0b78d1",
        "accent2": "#16a34a",
        "accent3": "#d63384",
    }
}

theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])
C = THEMES[theme]

# ---------------------------------------------------------------------
# Core CSS (typewriter, KPI rings, bars, glow)
# ---------------------------------------------------------------------
# caption text length is 96 chars — typewriter width uses that; cursor hides after typing
TYPEWRITER_CHARS = 96

st.markdown(f"""
<style>
/* Page base */
body {{
  background: {C['bg']};
  color: {C['text']};
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}}

/* Typewriter: types then cursor disappears */
@keyframes typing {{
  from {{ width: 0; }}
  to {{ width: {TYPEWRITER_CHARS}ch; }}
}}
@keyframes blinkCursor {{
  0% {{ border-right-color: {C['text']}; }}
  50% {{ border-right-color: transparent; }}
  100% {{ border-right-color: {C['text']}; }}
}}
@keyframes hideCursor {{
  from {{ border-right-color: {C['text']}; }}
  to {{ border-right-color: transparent; }}
}}
.typewriter {{
  display: inline-block;
  white-space: nowrap;
  overflow: hidden;
  width: {TYPEWRITER_CHARS}ch;
  border-right: 3px solid {C['text']};
  animation:
    typing 3.5s steps({TYPEWRITER_CHARS}, end),
    blinkCursor 0.8s step-end 3,
    hideCursor 0.3s ease forwards 4.3s;
  font-size: 18px;
  color: {C['muted']};
}}

/* KPI glass card */
.kpi-glass {{
  background: {C['card_bg']};
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  padding: 18px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  text-align: center;
}}

/* KPI ring container */
.kpi-ring {{
  width: 140px;
  height: 140px;
  border-radius: 50%;
  margin: 12px auto;
  position: relative;
  --value: 0;
  transition: --value 1s ease;
}}
.kpi-ring .ring-bg {{
  position: absolute; inset: 0;
  border-radius: 50%;
  background: conic-gradient(var(--color) calc(var(--value) * 1%), rgba(255,255,255,0.06) 0%);
  display:flex;align-items:center;justify-content:center;
  box-shadow: 0 6px 18px rgba(0,0,0,0.4);
}}
.kpi-ring .ring-inner {{
  width: 96px; height:96px; border-radius:50%;
  background: rgba(0,0,0,0.55);
  display:flex;align-items:center;justify-content:center;
  color: #fff; font-weight:700; font-size:20px;
  border: 2px solid rgba(255,255,255,0.04);
}}

/* small label */
.kpi-label {{ color: {C['muted']}; margin-top:8px; }}

/* Animated metric bars */
.metric-card {{
  background: {C['card_bg']};
  backdrop-filter: blur(8px);
  padding:12px;
  border-radius:10px;
  border:1px solid rgba(255,255,255,0.06);
  box-shadow: 0 6px 16px rgba(0,0,0,0.2);
}}
.metric-name {{ font-weight:700; color:{C['text']}; margin-bottom:6px; }}
.metric-value {{ font-weight:700; color:{C['muted']}; margin-left:8px; }}

/* bar */
.metric-bar {{
  height: 14px; border-radius:8px; background: rgba(0,0,0,0.15); overflow:hidden;
}}
.metric-bar-fill {{
  height:100%;
  width: 0%;
  border-radius:8px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.2) inset;
  transition: width 1.4s cubic-bezier(.2,.9,.2,1), box-shadow 0.4s;
}}
.metric-bar-fill.end-glow {{
  box-shadow: 0 8px 30px var(--glow-color);
}}

/* compact table styling override */
.dataframe tbody tr td {{ color: {C['text']}; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Header (Lottie + title + typewriter caption)
# ---------------------------------------------------------------------
col_h1, col_h2 = st.columns([1, 4])
with col_h1:
    if header_lottie:
        st_lottie(header_lottie, height=120)
with col_h2:
    st.markdown(f"<h1 style='color:{C['text']}; margin:0;'>Neural Translation Evaluation & Insights Platform</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='typewriter'>AI-driven analytics for benchmarking translation quality, coherence, and linguistic fidelity.</div>", unsafe_allow_html=True)

st.write("---")

# ---------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------
text = st.text_area("Enter text to evaluate:", height=140)
# single Run button (per your request)
run_clicked = st.button("Run Evaluation")

# ---------------------------------------------------------------------
# Utility metrics function
# ---------------------------------------------------------------------
def get_metrics(src, out):
    bleu = compute_bleu(src, out)
    efc = compute_efc(src, out)
    halluc = round(1 - efc, 3)
    semantic = round((bleu + efc) / 2, 3)
    return {
        "BLEU": float(np.clip(bleu, 0.0, 1.0)),
        "EFC": float(np.clip(efc, 0.0, 1.0)),
        "Hallucination": float(np.clip(halluc, 0.0, 1.0)),
        "Semantic": float(np.clip(semantic, 0.0, 1.0))
    }

# ---------------------------------------------------------------------
# When Run is clicked: show only loading animation (no spinner text),
# compute metrics, then render the animated dashboard and inject JS to animate
# ---------------------------------------------------------------------
if run_clicked:
    if not text.strip():
        st.error("Please enter text to evaluate.")
    else:
        # show only loading lottie
        loader = st.empty()
        with loader:
            if loading_animation:
                st_lottie(loading_animation, height=200)
            else:
                st.markdown("<div style='padding:20px; text-align:center;'>Loading...</div>", unsafe_allow_html=True)

        # simulate short delay for smoothness (if your models are quick you can reduce this)
        time.sleep(1.2)

        # run models (hidden outputs)
        out_b = baseline_translate(text)
        out_e = eact_translate(text)
        out_r = rgcld_translate(text)

        # compute metrics
        mB = get_metrics(text, out_b)
        mE = get_metrics(text, out_e)
        mR = get_metrics(text, out_r)

        # remove loader
        loader.empty()

        # TAB NAMES (enterprise)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive KPI Deck",
            "🧭 Model Trajectory Space (3D)",
            "📌 Multi-Metric Radar Insights",
            "📍 Error & Similarity Analytics",
            "📑 Model Scorecard Matrix"
        ])

        # -------------------------
        # Tab 1: KPI Deck (animated rings)
        # -------------------------
        with tab1:
            cols = st.columns(3)
            items = [
                ("Baseline", mB["BLEU"], C["accent1"], "BLEU"),
                ("EACT",     mE["BLEU"], C["accent2"], "BLEU"),
                ("RG-CLD",   mR["BLEU"], C["accent3"], "BLEU"),
            ]
            # We'll render ring containers with data-target attributes and numeric spans
            for (title, val, color, label), col in zip(items, cols):
                pct = round(float(val) * 100, 2)
                # render HTML card
                col.markdown(f"""
                <div class="kpi-glass">
                  <div style="font-size:14px;color:{color};font-weight:700;margin-bottom:6px">{title}</div>
                  <div class="kpi-ring" data-target="{pct}" data-color="{color}">
                    <div class="ring-bg" style="--color:{color}; --value:0;"></div>
                    <div class="ring-inner">
                      <span class="kpi-number">0</span>
                    </div>
                  </div>
                  <div class="kpi-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        # -------------------------
        # Tab 2: 3D BLEU–EFC line (keeps the enhanced line)
        # -------------------------
        with tab2:
            st.markdown("### 3D BLEU–EFC Metric Trajectory (Gradient Line)")
            X = [mB["BLEU"], mE["BLEU"], mR["BLEU"]]
            Y = [mB["EFC"],  mE["EFC"],  mR["EFC"]]
            Z = [0.0, 0.5, 1.0]
            labels = ["Baseline", "EACT", "RG-CLD"]

            t_orig = np.linspace(0, 1, len(X))
            t_fine = np.linspace(0, 1, 300)
            xs = np.interp(t_fine, t_orig, X)
            ys = np.interp(t_fine, t_orig, Y)
            zs = np.interp(t_fine, t_orig, Z)

            fig3d = go.Figure()
            fig3d.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(width=8, color=xs, colorscale='Turbo', dash='dot'),
                name='Trajectory'
            ))
            fig3d.add_trace(go.Scatter3d(
                x=X, y=Y, z=Z,
                mode='markers+text',
                marker=dict(size=8, color=[C["accent1"], C["accent2"], C["accent3"]], line=dict(width=2, color='white')),
                text=labels,
                textposition='top center',
                name='Models'
            ))
            # subtle ground surface
            xx, yy = np.meshgrid(np.linspace(0,1,6), np.linspace(0,1,6))
            fig3d.add_trace(go.Surface(x=xx, y=yy, z=np.zeros_like(xx), showscale=False, opacity=0.08, colorscale=[[0,'rgba(100,100,100,0.12)'],[1,'rgba(200,200,200,0.02)']]))
            fig3d.update_layout(scene=dict(xaxis=dict(title='BLEU', range=[0,1]), yaxis=dict(title='EFC', range=[0,1]), zaxis=dict(title='Depth', showticklabels=False)), height=640, margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
            st.plotly_chart(fig3d, use_container_width=True)

        # -------------------------
        # Tab 3: Radar Insights
        # -------------------------
        with tab3:
            cats = ["BLEU", "EFC", "Hallucination", "Semantic"]
            figR = go.Figure()
            figR.add_trace(go.Scatterpolar(r=[mB[c] for c in cats], theta=cats, fill='toself', name='Baseline'))
            figR.add_trace(go.Scatterpolar(r=[mE[c] for c in cats], theta=cats, fill='toself', name='EACT'))
            figR.add_trace(go.Scatterpolar(r=[mR[c] for c in cats], theta=cats, fill='toself', name='RG-CLD'))
            figR.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=620)
            st.plotly_chart(figR, use_container_width=True)

        # -------------------------
        # Tab 4: Advanced animated bars (Error & Similarity Analytics)
        # -------------------------
        with tab4:
            row1, row2 = st.columns([2,2])
            # Hallucination bars
            with row1:
                st.markdown("<h4 style='margin-top:0;'>Hallucination Rate (Lower = Better)</h4>", unsafe_allow_html=True)
                bars = [
                    ("Baseline", mB["Hallucination"], "#ff4e50"),
                    ("EACT", mE["Hallucination"], "#ffa600"),
                    ("RG-CLD", mR["Hallucination"], "#ff2a68"),
                ]
                for name, val, color in bars:
                    pct = round(val * 100, 2)
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:10px;">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div class="metric-name">{name}</div>
                        <div class="metric-value"><span class="bar-number">0</span>%</div>
                      </div>
                      <div class="metric-bar" data-target="{pct}" data-glow="{color}">
                        <div class="metric-bar-fill" style="--glow:{color};"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Semantic similarity bars
            with row2:
                st.markdown("<h4 style='margin-top:0;'>Semantic Similarity (Higher = Better)</h4>", unsafe_allow_html=True)
                bars2 = [
                    ("Baseline", mB["Semantic"], "#30cfd0"),
                    ("EACT", mE["Semantic"], "#6a5acd"),
                    ("RG-CLD", mR["Semantic"], "#4facfe"),
                ]
                for name, val, color in bars2:
                    pct = round(val * 100, 2)
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:10px;">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div class="metric-name">{name}</div>
                        <div class="metric-value"><span class="bar-number">0</span>%</div>
                      </div>
                      <div class="metric-bar" data-target="{pct}" data-glow="{color}">
                        <div class="metric-bar-fill" style="--glow:{color};"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # -------------------------
        # Tab 5: Scorecard table
        # -------------------------
        with tab5:
            st.write("### Model Scorecard Matrix")
            st.table({
                "Model": ["Baseline", "EACT", "RG-CLD"],
                "BLEU": [mB["BLEU"], mE["BLEU"], mR["BLEU"]],
                "EFC": [mB["EFC"], mE["EFC"], mR["EFC"]],
                "Hallucination": [mB["Hallucination"], mE["Hallucination"], mR["Hallucination"]],
                "Semantic": [mB["Semantic"], mE["Semantic"], mR["Semantic"]],
            })

        # ---------------------------------------------------------------------
        # Inject JavaScript to animate KPI rings and bars + counting numbers
        # We'll use streamlit.components.v1.html to safely run JS in the page.
        # ---------------------------------------------------------------------
        animate_js = f"""
        <script>
        (function(){{

            // animate KPI rings: fill and number counting
            const rings = Array.from(document.querySelectorAll('.kpi-ring'));
            rings.forEach(r => {{
                const target = parseFloat(r.getAttribute('data-target')) || 0;
                const color = r.getAttribute('data-color') || '{C['accent1']}';
                const ringBg = r.querySelector('.ring-bg');
                const num = r.querySelector('.kpi-number');

                // set CSS variable color
                ringBg.style.setProperty('--color', color);
                // animate fill: use requestAnimationFrame to increment
                let start = null;
                const duration = 1100;
                function step(timestamp) {{
                    if (!start) start = timestamp;
                    const progress = Math.min((timestamp - start) / duration, 1);
                    const current = progress * target;
                    ringBg.style.background = `conic-gradient(${{color}} ${{current}}%, rgba(255,255,255,0.06) 0%)`;
                    num.innerText = current.toFixed(2);
                    if (progress < 1) {{
                        window.requestAnimationFrame(step);
                    }} else {{
                        // final set
                        ringBg.style.background = `conic-gradient(${{color}} ${{target}}%, rgba(255,255,255,0.06) 0%)`;
                        num.innerText = target.toFixed(2);
                    }}
                }}
                window.requestAnimationFrame(step);
            }});

            // animate bars: expand width and count numbers
            const bars = Array.from(document.querySelectorAll('.metric-bar'));
            bars.forEach(bar => {{
                const target = parseFloat(bar.getAttribute('data-target')) || 0;
                const glow = bar.getAttribute('data-glow') || '#4facfe';
                const fill = bar.querySelector('.metric-bar-fill');
                const numberSpan = bar.parentElement.querySelector('.bar-number') || null;

                // animate width
                setTimeout(() => {{
                    fill.style.width = target + '%';
                    fill.style.background = glow;
                    fill.classList.add('end-glow');
                    fill.style.setProperty('--glow-color', glow + '55');
                }}, 50);

                // count-up number
                if (numberSpan) {{
                    let start = null;
                    const duration = 1100;
                    function countStep(ts) {{
                        if (!start) start = ts;
                        const prog = Math.min((ts - start) / duration, 1);
                        const cur = Math.floor(prog * target);
                        numberSpan.innerText = cur;
                        if (prog < 1) {{
                            window.requestAnimationFrame(countStep);
                        }} else {{
                            numberSpan.innerText = Math.round(target);
                        }}
                    }}
                    window.requestAnimationFrame(countStep);
                }}
            }});

        }})();
        </script>
        """
        components.html(animate_js, height=10, scrolling=False)
