# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
import time
import streamlit.components.v1 as components

# Import your model wrappers and scoring utils (must exist in your repo)
from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate
from utils.scoring import compute_bleu, compute_efc

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Neural Translation Evaluation & Insights Platform", layout="wide")

# -------------------------
# Helpers
# -------------------------
def load_lottie(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

loading_animation = load_lottie("https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json")
header_lottie = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")

# -------------------------
# Theme / Colors
# -------------------------
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

# -------------------------
# Base CSS (keeps Streamlit layout consistent)
# -------------------------
TYPEWRITER_CHARS = 96
st.markdown(f"""
<style>
/* Page base */
body {{
  background: {C['bg']};
  color: {C['text']};
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}}

/* Typewriter animation (cursor disappears after typing) */
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

/* keep Streamlit table/text colors readable */
.stMarkdown, .css-1d391kg p, .stText {{
  color: {C['text']};
}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header (Lottie + Title + Caption)
# -------------------------
col1, col2 = st.columns([1, 4])
with col1:
    if header_lottie:
        st_lottie(header_lottie, height=120)
with col2:
    st.markdown(f"<h1 style='color:{C['text']}; margin:0;'>Neural Translation Evaluation & Insights Platform</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='typewriter'>AI-driven analytics for benchmarking translation quality, coherence, and linguistic fidelity.</div>", unsafe_allow_html=True)

st.write("---")

# -------------------------
# Input
# -------------------------
text = st.text_area("Enter source text to evaluate:", height=140)
run_clicked = st.button("Run Evaluation")

# -------------------------
# Metric helper
# -------------------------
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

# -------------------------
# When run: compute & render
# -------------------------
if run_clicked:
    if not text.strip():
        st.error("Please enter text to evaluate.")
    else:
        # Show only the loading Lottie animation (no spinner text)
        loader = st.empty()
        with loader:
            if loading_animation:
                st_lottie(loading_animation, height=200)
            else:
                st.markdown("<div style='padding:20px; text-align:center;'>Loading...</div>", unsafe_allow_html=True)

        # small delay for smoothness (optional)
        time.sleep(0.8)

        # Run model translations (these should be implemented to return translated text)
        out_b = baseline_translate(text)
        out_e = eact_translate(text)
        out_r = rgcld_translate(text)

        # Compute metrics
        mB = get_metrics(text, out_b)
        mE = get_metrics(text, out_e)
        mR = get_metrics(text, out_r)

        # Clear loader
        loader.empty()

        # Tab names (enterprise)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive KPI Deck",
            "🧭 Model Trajectory Space (3D)",
            "📌 Multi-Metric Radar Insights",
            "📍 Error & Similarity Analytics",
            "📑 Model Scorecard Matrix"
        ])

        # -------------------------
        # Tab 1: KPI Deck — rendered via components.html (self-contained)
        # Animated rings + numeric counters live correctly on Streamlit Cloud
        # -------------------------
        with tab1:
            # Prepare metrics for embedding
            kpi_items = [
                {"title": "Baseline", "value": round(mB["BLEU"] * 100, 2), "color": C["accent1"], "label": "BLEU (%)"},
                {"title": "EACT",     "value": round(mE["BLEU"] * 100, 2), "color": C["accent2"], "label": "BLEU (%)"},
                {"title": "RG-CLD",   "value": round(mR["BLEU"] * 100, 2), "color": C["accent3"], "label": "BLEU (%)"},
            ]

            kpi_html = f"""
            <div style="display:flex; gap:20px; justify-content:space-between; align-items:flex-start;">
            """
            for it in kpi_items:
                kpi_html += f"""
                <div style="flex:1; max-width:320px;">
                  <div style="background:{C['card_bg']}; padding:18px; border-radius:14px; text-align:center; border:1px solid rgba(255,255,255,0.06);">
                    <div style="font-size:14px; color:{it['color']}; font-weight:700; margin-bottom:8px;">{it['title']}</div>
                    <div class="kpi-ring" data-target="{it['value']}" data-color="{it['color']}" style="width:140px;height:140px;border-radius:50%;margin:0 auto;position:relative;">
                      <div class="ring-bg" style="position:absolute; inset:0; border-radius:50%; display:flex; align-items:center; justify-content:center; box-shadow:0 6px 18px rgba(0,0,0,0.4); background: conic-gradient({it['color']} 0%, rgba(255,255,255,0.06) 0%);"></div>
                      <div style="width:96px;height:96px;border-radius:50%; background: rgba(0,0,0,0.55); display:flex;align-items:center;justify-content:center; color:white; font-weight:700; font-size:20px; border:2px solid rgba(255,255,255,0.04);">
                        <span class="kpi-number">0.00</span>
                      </div>
                    </div>
                    <div style="margin-top:8px; color:{C['muted']};">{it['label']}</div>
                  </div>
                </div>
                """

            kpi_html += "</div>"

            # JS to animate rings (runs inside iframe created by components.html - safe)
            kpi_js = f"""
            <script>
            (function(){{
                const rings = document.querySelectorAll('.kpi-ring');
                rings.forEach(r => {{
                    const target = parseFloat(r.getAttribute('data-target')) || 0;
                    const color = r.getAttribute('data-color') || '{C['accent1']}';
                    const bg = r.querySelector('.ring-bg');
                    const num = r.querySelector('.kpi-number');

                    // animate using requestAnimationFrame
                    let start = null;
                    const duration = 1100;
                    function step(ts) {{
                        if (!start) start = ts;
                        const progress = Math.min((ts - start) / duration, 1);
                        const current = progress * target;
                        bg.style.background = `conic-gradient(${color} ${current}%, rgba(255,255,255,0.06) 0%)`;
                        num.innerText = current.toFixed(2);
                        if (progress < 1) {{
                            window.requestAnimationFrame(step);
                        }} else {{
                            bg.style.background = `conic-gradient(${color} ${target}%, rgba(255,255,255,0.06) 0%)`;
                            num.innerText = target.toFixed(2);
                        }}
                    }}
                    window.requestAnimationFrame(step);
                }});
            }})();
            </script>
            """

            components.html(kpi_html + kpi_js, height=260, scrolling=False)

        # -------------------------
        # Tab 2: 3D BLEU–EFC Plotly chart
        # -------------------------
        with tab2:
            st.markdown("### 3D BLEU–EFC Metric Trajectory")
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
                text=labels, textposition='top center', name='Models'
            ))
            xx, yy = np.meshgrid(np.linspace(0,1,6), np.linspace(0,1,6))
            fig3d.add_trace(go.Surface(x=xx, y=yy, z=np.zeros_like(xx), showscale=False, opacity=0.08, colorscale=[[0,'rgba(100,100,100,0.12)'],[1,'rgba(200,200,200,0.02)']]))
            fig3d.update_layout(scene=dict(xaxis=dict(title='BLEU', range=[0,1]), yaxis=dict(title='EFC', range=[0,1]), zaxis=dict(title='Depth', showticklabels=False)), height=640, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
            st.plotly_chart(fig3d, use_container_width=True)

        # -------------------------
        # Tab 3: Radar (Plotly)
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
        # Tab 4: Animated Bars — render via components.html (self-contained)
        # This ensures animation JS runs reliably on Streamlit Cloud
        # -------------------------
        with tab4:
            # prepare bars data (Hallucination & Semantic)
            halluc = [
                {"name": "Baseline", "value": round(mB["Hallucination"] * 100, 2), "color": "#ff4e50"},
                {"name": "EACT",     "value": round(mE["Hallucination"] * 100, 2), "color": "#ffa600"},
                {"name": "RG-CLD",   "value": round(mR["Hallucination"] * 100, 2), "color": "#ff2a68"},
            ]
            semantic = [
                {"name": "Baseline", "value": round(mB["Semantic"] * 100, 2), "color": "#30cfd0"},
                {"name": "EACT",     "value": round(mE["Semantic"] * 100, 2), "color": "#6a5acd"},
                {"name": "RG-CLD",   "value": round(mR["Semantic"] * 100, 2), "color": "#4facfe"},
            ]

            bars_html = "<div style='display:flex; gap:20px; justify-content:space-between;'>"

            # left column (hallucination)
            bars_html += "<div style='flex:1; max-width:48%;'>"
            bars_html += f"<h4 style='margin-top:0;color:{C['text']}'>Hallucination Rate (Lower = Better)</h4>"
            for b in halluc:
                bars_html += f"""
                <div style='background:{C['card_bg']}; padding:12px; border-radius:10px; margin-bottom:10px; border:1px solid rgba(255,255,255,0.06)'>
                  <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div style='font-weight:700; color:{C['text']}'>{b['name']}</div>
                    <div style='font-weight:700; color:{C['muted']}'><span class='bar-number'>0</span>%</div>
                  </div>
                  <div class='metric-bar' data-target='{b['value']}' data-glow='{b['color']}' style='height:14px; border-radius:8px; background:rgba(0,0,0,0.12); overflow:hidden; margin-top:8px;'>
                    <div class='metric-bar-fill' style='width:0%; height:100%; border-radius:8px;'></div>
                  </div>
                </div>
                """

            bars_html += "</div>"

            # right column (semantic)
            bars_html += "<div style='flex:1; max-width:48%;'>"
            bars_html += f"<h4 style='margin-top:0;color:{C['text']}'>Semantic Similarity (Higher = Better)</h4>"
            for b in semantic:
                bars_html += f"""
                <div style='background:{C['card_bg']}; padding:12px; border-radius:10px; margin-bottom:10px; border:1px solid rgba(255,255,255,0.06)'>
                  <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div style='font-weight:700; color:{C['text']}'>{b['name']}</div>
                    <div style='font-weight:700; color:{C['muted']}'><span class='bar-number'>0</span>%</div>
                  </div>
                  <div class='metric-bar' data-target='{b['value']}' data-glow='{b['color']}' style='height:14px; border-radius:8px; background:rgba(0,0,0,0.12); overflow:hidden; margin-top:8px;'>
                    <div class='metric-bar-fill' style='width:0%; height:100%; border-radius:8px;'></div>
                  </div>
                </div>
                """

            bars_html += "</div></div>"

            # JS to animate bars and update numbers
            bars_js = """
            <script>
            (function(){
                const bars = Array.from(document.querySelectorAll('.metric-bar'));
                bars.forEach(bar => {
                    const target = parseFloat(bar.getAttribute('data-target')) || 0;
                    const glow = bar.getAttribute('data-glow') || '#4facfe';
                    const fill = bar.querySelector('.metric-bar-fill');
                    const numberSpan = bar.parentElement.querySelector('.bar-number');

                    // animate width
                    setTimeout(() => {
                        fill.style.transition = 'width 1.4s cubic-bezier(.2,.9,.2,1)';
                        fill.style.width = target + '%';
                        fill.style.background = glow;
                        // glow effect via box-shadow
                        fill.style.boxShadow = `0 8px 30px ${glow}55`;
                    }, 60);

                    // count up number
                    if (numberSpan) {
                        let start = null;
                        const duration = 1100;
                        function step(ts) {
                            if (!start) start = ts;
                            const prog = Math.min((ts - start) / duration, 1);
                            const cur = Math.round(prog * target);
                            numberSpan.innerText = cur;
                            if (prog < 1) {
                                window.requestAnimationFrame(step);
                            } else {
                                numberSpan.innerText = Math.round(target);
                            }
                        }
                        window.requestAnimationFrame(step);
                    }
                });
            })();
            </script>
            """

            components.html(bars_html + bars_js, height=520, scrolling=True)

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
