# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
import time
import streamlit.components.v1 as components

# Import your model wrappers & scoring utils
from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate
from utils.scoring import compute_bleu, compute_efc


# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Neural Translation Evaluation & Insights Platform",
    layout="wide"
)


# ---------------------------------------------------------------
# Load Lottie animations
# ---------------------------------------------------------------
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None


loading_animation = load_lottie("https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json")
header_lottie = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")


# ---------------------------------------------------------------
# THEMES
# ---------------------------------------------------------------
THEMES = {
    "Dark": {
        "bg": "linear-gradient(135deg, #0a0f1f, #151c2f)",
        "card_bg": "rgba(255,255,255,0.07)",
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


# ---------------------------------------------------------------
# CSS  (Typewriter + Layout + Base Styling)
# ---------------------------------------------------------------
TYPEWRITER_CHARS = 96

st.markdown(f"""
<style>

body {{
  background: {C['bg']};
  color: {C['text']};
  font-family: "Segoe UI", Roboto, sans-serif;
}}

/* Typewriter */
@keyframes typing {{
  from {{ width: 0; }}
  to {{ width: {TYPEWRITER_CHARS}ch; }}
}}

@keyframes blink {{
  50% {{ border-color: transparent; }}
}}

@keyframes hideCursor {{
  from {{ border-right-color:{C['text']}; }}
  to {{ border-right-color: transparent; }}
}}

.typewriter {{
  display:inline-block;
  overflow:hidden;
  white-space:nowrap;
  border-right:3px solid {C['text']};
  width:{TYPEWRITER_CHARS}ch;
  animation: typing 3.5s steps({TYPEWRITER_CHARS}, end), 
             blink .75s step-end 3,
             hideCursor .4s ease forwards 4.3s;
  font-size:18px;
  color:{C['muted']};
}}

/* KPI RING CARDS */
.kpi-card {{
  background:{C['card_bg']};
  padding:18px;
  border-radius:14px;
  border:1px solid rgba(255,255,255,0.08);
  box-shadow:0 6px 18px rgba(0,0,0,0.35);
  text-align:center;
}}

/* TABLE COLOR FIX */
td, th, .stMarkdown p {{
  color: {C['text']} !important;
}}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
h1c1, h1c2 = st.columns([1, 4])

with h1c1:
    if header_lottie:
        st_lottie(header_lottie, height=120)

with h1c2:
    st.markdown(f"<h1>Neural Translation Evaluation & Insights Platform</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='typewriter'>AI-driven analytics for benchmarking translation quality, coherence, and linguistic fidelity.</div>",
        unsafe_allow_html=True
    )

st.write("---")


# ---------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------
text = st.text_area("Enter text to evaluate:", height=140)
run_clicked = st.button("Run Evaluation")


# ---------------------------------------------------------------
# Compute metrics helper
# ---------------------------------------------------------------
def get_metrics(src, out):
    bleu = compute_bleu(src, out)
    efc = compute_efc(src, out)
    halluc = round(1 - efc, 3)
    semantic = round((bleu + efc) / 2, 3)
    return {
        "BLEU": float(np.clip(bleu, 0, 1)),
        "EFC": float(np.clip(efc, 0, 1)),
        "Hallucination": float(np.clip(halluc, 0, 1)),
        "Semantic": float(np.clip(semantic, 0, 1))
    }


# ---------------------------------------------------------------
# RUN
# ---------------------------------------------------------------
if run_clicked:

    if not text.strip():
        st.error("Please enter text!")
    else:

        # Loading only animation, no text
        loader = st.empty()
        with loader:
            if loading_animation:
                st_lottie(loading_animation, height=200)

        time.sleep(3.2)

        # Get outputs
        out_b = baseline_translate(text)
        out_e = eact_translate(text)
        out_r = rgcld_translate(text)

        # Compute metrics
        mB = get_metrics(text, out_b)
        mE = get_metrics(text, out_e)
        mR = get_metrics(text, out_r)

        loader.empty()

        # -------------------------------------------------------
        # TABS
        # -------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive KPI Deck",
            "🧭 Model Trajectory Space (3D)",
            "📌 Multi-Metric Radar Insights",
            "📍 Error & Similarity Analytics",
            "📑 Model Scorecard Matrix"
        ])

        # -------------------------------------------------------
        # TAB 1 – KPI RINGS ★ Animated
        # -------------------------------------------------------
        with tab1:

            kpis = [
                ("Baseline", mB["BLEU"] * 100, C["accent1"]),
                ("EACT", mE["BLEU"] * 100, C["accent2"]),
                ("RG-CLD", mR["BLEU"] * 100, C["accent3"]),
            ]

            html = "<div style='display:flex; gap:20px;'>"

            for title, val, color in kpis:
                html += f"""
                <div class="kpi-card" style="flex:1;">
                    <div style='font-size:14px;color:{color};font-weight:700;'>{title}</div>
                    <div class="kpi-ring" data-target="{val}" data-color="{color}" 
                        style="width:140px;height:140px;border-radius:50%; margin:10px auto; position:relative;">
                        
                        <div class="ring-bg" 
                             style="position:absolute; inset:0; border-radius:50%;
                             background:conic-gradient({color} 0%, rgba(255,255,255,0.06) 0%); 
                             display:flex; align-items:center; justify-content:center;">
                        </div>

                        <div style="width:96px;height:96px;background:rgba(0,0,0,0.55);border-radius:50%;
                                    position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                                    display:flex;align-items:center;justify-content:center;
                                    color:white;font-size:20px;font-weight:700;">
                            <span class="kpi-number">0.00</span>
                        </div>
                    </div>

                    <div style="color:{C['muted']}; margin-top:6px;">BLEU (%)</div>
                </div>
                """

            html += "</div>"

            # KPI JS — FIXED (no backticks)
            js = """
            <script>
            (function(){
                const rings = document.querySelectorAll('.kpi-ring');

                rings.forEach(function(r){
                    const target = parseFloat(r.getAttribute('data-target')) || 0;
                    const color = r.getAttribute('data-color') || '#4facfe';
                    const bg = r.querySelector('.ring-bg');
                    const num = r.querySelector('.kpi-number');

                    let start = null;
                    const duration = 1000;

                    function step(ts){
                        if(!start) start = ts;
                        const progress = Math.min((ts - start) / duration, 1);
                        const current = progress * target;

                        bg.style.background =
                            "conic-gradient(" + color + " " + current + "%, rgba(255,255,255,0.06) 0%)";

                        num.innerText = current.toFixed(2);

                        if(progress < 1){
                            window.requestAnimationFrame(step);
                        } else {
                            bg.style.background =
                                "conic-gradient(" + color + " " + target + "%, rgba(255,255,255,0.06) 0%)";
                            num.innerText = target.toFixed(2);
                        }
                    }

                    window.requestAnimationFrame(step);
                });
            })();
            </script>
            """

            components.html(html + js, height=300, scrolling=False)

        # -------------------------------------------------------
        # TAB 2 – 3D Plot
        # -------------------------------------------------------
        with tab2:

            st.markdown("### 3D BLEU–EFC Trajectory")

            X = [mB["BLEU"], mE["BLEU"], mR["BLEU"]]
            Y = [mB["EFC"],  mE["EFC"],  mR["EFC"]]
            Z = [0.0, 0.5, 1.0]

            t = np.linspace(0, 1, 300)
            xs = np.interp(t, [0, 0.5, 1], X)
            ys = np.interp(t, [0, 0.5, 1], Y)
            zs = np.interp(t, [0, 0.5, 1], Z)

            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(width=8, color=xs, colorscale="Turbo", dash="dot")
            ))
            fig.add_trace(go.Scatter3d(
                x=X, y=Y, z=Z, mode="markers+text",
                marker=dict(size=8, color=[C["accent1"],C["accent2"],C["accent3"]],
                            line=dict(width=2,color="white")),
                text=["Baseline","EACT","RG-CLD"], textposition="top center"
            ))

            fig.update_layout(height=650, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------
        # TAB 3 – RADAR
        # -------------------------------------------------------
        with tab3:

            cats = ["BLEU", "EFC", "Hallucination", "Semantic"]

            figR = go.Figure()
            figR.add_trace(go.Scatterpolar(r=[mB[c] for c in cats], theta=cats, fill="toself", name="Baseline"))
            figR.add_trace(go.Scatterpolar(r=[mE[c] for c in cats], theta=cats, fill="toself", name="EACT"))
            figR.add_trace(go.Scatterpolar(r=[mR[c] for c in cats], theta=cats, fill="toself", name="RG-CLD"))

            figR.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=620)
            st.plotly_chart(figR, use_container_width=True)

        # -------------------------------------------------------
        # TAB 4 – Animated Bars
        # -------------------------------------------------------
        with tab4:

            halluc = [
                ("Baseline", mB["Hallucination"] * 100, "#ff4e50"),
                ("EACT", mE["Hallucination"] * 100, "#ffa600"),
                ("RG-CLD", mR["Hallucination"] * 100, "#ff2a68")
            ]
            semantic = [
                ("Baseline", mB["Semantic"] * 100, "#30cfd0"),
                ("EACT", mE["Semantic"] * 100, "#6a5acd"),
                ("RG-CLD", mR["Semantic"] * 100, "#4facfe")
            ]

            html = "<div style='display:flex; gap:25px;'>"

            # Hallucination
            html += "<div style='flex:1;'><h4>Hallucination Rate</h4>"
            for name, val, color in halluc:

                html += f"""
                <div style="background:{C['card_bg']}; padding:12px; border-radius:10px;
                            margin-bottom:10px; border:1px solid rgba(255,255,255,0.06);">
                    <div style="display:flex; justify-content:space-between;">
                        <div style="color:{C['text']};font-weight:700">{name}</div>
                        <div style="color:{C['muted']};font-weight:700"><span class='bar-num'>0</span>%</div>
                    </div>

                    <div class="bar-wrap" data-target="{val}" data-color="{color}"
                         style="height:14px; background:rgba(0,0,0,0.15); border-radius:8px; margin-top:8px;">
                        <div class="bar-fill" style="width:0%; height:100%; 
                            background:{color}; border-radius:8px;">
                        </div>
                    </div>
                </div>
                """
            html += "</div>"

            # Semantic
            html += "<div style='flex:1;'><h4>Semantic Similarity</h4>"
            for name, val, color in semantic:

                html += f"""
                <div style="background:{C['card_bg']}; padding:12px; border-radius:10px;
                            margin-bottom:10px; border:1px solid rgba(255,255,255,0.06);">
                    <div style="display:flex; justify-content:space-between;">
                        <div style="color:{C['text']};font-weight:700">{name}</div>
                        <div style="color:{C['muted']};font-weight:700"><span class='bar-num'>0</span>%</div>
                    </div>

                    <div class="bar-wrap" data-target="{val}" data-color="{color}"
                         style="height:14px; background:rgba(0,0,0,0.15); border-radius:8px; margin-top:8px;">
                        <div class="bar-fill" style="width:0%; height:100%; 
                            background:{color}; border-radius:8px;">
                        </div>
                    </div>
                </div>
                """
            html += "</div></div>"

            # JS — FIXED VERSION
            js = """
            <script>
            (function(){
                const bars = document.querySelectorAll('.bar-wrap');

                bars.forEach(function(bar){
                    const target = parseFloat(bar.getAttribute('data-target')) || 0;
                    const color = bar.getAttribute('data-color') || '#4facfe';
                    const fill = bar.querySelector('.bar-fill');
                    const num = bar.parentElement.querySelector('.bar-num');

                    setTimeout(function(){
                        fill.style.transition = 'width 1.4s cubic-bezier(.2,.9,.2,1)';
                        fill.style.width = target + '%';
                        fill.style.background = color;
                        fill.style.boxShadow = "0 8px 30px " + color + "55";
                    }, 80);

                    let start = null;
                    const duration = 1000;

                    function countStep(ts){
                        if(!start) start = ts;
                        const prog = Math.min((ts - start)/duration, 1);
                        const cur = Math.round(prog * target);
                        num.innerText = cur;
                        if(prog < 1){
                            window.requestAnimationFrame(countStep);
                        } else {
                            num.innerText = Math.round(target);
                        }
                    }
                    window.requestAnimationFrame(countStep);

                });
            })();
            </script>
            """

            components.html(html + js, height=550, scrolling=False)

        # -------------------------------------------------------
        # TAB 5 – TABLE
        # -------------------------------------------------------
        with tab5:
            st.write("### Model Scorecard Matrix")
            st.table({
                "Model": ["Baseline", "EACT", "RG-CLD"],
                "BLEU": [mB["BLEU"], mE["BLEU"], mR["BLEU"]],
                "EFC": [mB["EFC"], mE["EFC"], mR["EFC"]],
                "Hallucination": [mB["Hallucination"], mE["Hallucination"], mR["Hallucination"]],
                "Semantic": [mB["Semantic"], mE["Semantic"], mR["Semantic"]],
            })
