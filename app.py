import streamlit as st
import matplotlib.pyplot as plt

from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate

from utils.scoring import compute_bleu, compute_efc

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="Next-Level MT Dashboard", layout="wide")

# --------------------------------------------------------------
# CUSTOM CSS FOR ANIMATIONS + PREMIUM UI
# --------------------------------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f0f0f, #1a1a1a);
    color: #fff;
}

/* Centered Title */
h1 {
    text-align: center;
    color: #fff;
}

/* Glass Card Panel */
.glass {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.12);
    transition: transform 0.3s;
}

.glass:hover {
    transform: translateY(-5px);
}

/* -------------------
   ANIMATED KPI CIRCLES
---------------------- */

.kpi-circle {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    background: conic-gradient(var(--color) calc(var(--value) * 1%), #333 0%);
    display: flex;
    justify-content: center;
    align-items: center;
    margin: auto;
    box-shadow: 0 0 20px var(--color-glow);
    animation: fadeIn 1s ease forwards;
}

.kpi-circle-inner {
    width: 110px;
    height: 110px;
    border-radius: 50%;
    background: #0d0d0d;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 26px;
    font-weight: 700;
    color: #fff;
}

/* -------------------
   ANIMATED HORIZONTAL BARS
---------------------- */

.metric-bar {
    height: 18px;
    border-radius: 10px;
    background: #333;
    overflow: hidden;
    position: relative;
}

.metric-bar-fill {
    height: 100%;
    border-radius: 10px;
    animation: fillBar 1.8s ease forwards;
}

@keyframes fillBar {
    from { width: 0%; }
    to { width: var(--width); }
}

/* Count-Up Animation */
@keyframes countUp {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Fade In */
@keyframes fadeIn {
    from { opacity: 0; transform: scale(0.85); }
    to { opacity: 1; transform: scale(1); }
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# HEADER
# --------------------------------------------------------------
st.markdown("""
<h1>🚀 Ultra-Premium Animated MT Evaluation Dashboard</h1>
<p style="text-align:center; color:#bbb; font-size:18px;">
Benchmark reliability & factual consistency with visually stunning analytics.
</p>
<hr>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# INPUT BOX
# --------------------------------------------------------------
text = st.text_area("Enter Text for Evaluation:", height=120)

if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Please enter text first.")
    else:
        # ===============================
        # MODEL EXECUTION (NO DISPLAY)
        # ===============================
        baseline_out = baseline_translate(text)
        eact_out = eact_translate(text)
        rgcld_out = rgcld_translate(text)

        # ===============================
        # METRICS
        # ===============================
        def metrics(out):
            bleu = compute_bleu(text, out)
            efc = compute_efc(text, out)
            halluc = round(1 - efc, 3)
            semantic = round((bleu + efc) / 2, 3)
            return bleu, efc, halluc, semantic

        bleu_b, efc_b, hall_b, sem_b = metrics(baseline_out)
        bleu_e, efc_e, hall_e, sem_e = metrics(eact_out)
        bleu_r, efc_r, hall_r, sem_r = metrics(rgcld_out)

        # ===============================
        # KPI SECTION — ANIMATED CIRCLES
        # ===============================
        st.subheader("💎 Animated KPI Rings")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="glass">
                <h3 style="text-align:center;">Baseline</h3>
                <div class="kpi-circle" style="--value:{bleu_b*100}; --color:#4facfe; --color-glow:#4facfe55;">
                    <div class="kpi-circle-inner">{bleu_b}</div>
                </div>
                <p style="text-align:center; margin-top:10px;">BLEU Score</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="glass">
                <h3 style="text-align:center;">EACT</h3>
                <div class="kpi-circle" style="--value:{bleu_e*100}; --color:#43e97b; --color-glow:#43e97b55;">
                    <div class="kpi-circle-inner">{bleu_e}</div>
                </div>
                <p style="text-align:center; margin-top:10px;">BLEU Score</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="glass">
                <h3 style="text-align:center;">RG-CLD</h3>
                <div class="kpi-circle" style="--value:{bleu_r*100}; --color:#fa709a; --color-glow:#fa709a55;">
                    <div class="kpi-circle-inner">{bleu_r}</div>
                </div>
                <p style="text-align:center; margin-top:10px;">BLEU Score</p>
            </div>
            """, unsafe_allow_html=True)

        # ===============================
        # ANIMATED METRIC BARS
        # ===============================
        st.subheader("⚡ Animated Reliability Metrics")

        bar_col1, bar_col2 = st.columns(2)

        # HALLUCINATION RATE
        with bar_col1:
            st.markdown("<h4>Hallucination Rate</h4>", unsafe_allow_html=True)
            for model, val, color in [
                ("Baseline", hall_b, "#ff4e50"),
                ("EACT", hall_e, "#ff9500"),
                ("RG-CLD", hall_r, "#ff2a68")
            ]:
                st.markdown(f"""
                <p style="margin-bottom:5px;"><b>{model}</b> — {val}</p>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="--width:{val*100}%; background:{color};"></div>
                </div>
                <br>
                """, unsafe_allow_html=True)

        # SEMANTIC SIMILARITY
        with bar_col2:
            st.markdown("<h4>Semantic Similarity</h4>", unsafe_allow_html=True)
            for model, val, color in [
                ("Baseline", sem_b, "#30cfd0"),
                ("EACT", sem_e, "#8360c3"),
                ("RG-CLD", sem_r, "#4facfe")
            ]:
                st.markdown(f"""
                <p style="margin-bottom:5px;"><b>{model}</b> — {val}</p>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="--width:{val*100}%; background:{color};"></div>
                </div>
                <br>
                """, unsafe_allow_html=True)

        # ===============================
        # COMPARISON TABLE
        # ===============================
        st.subheader("📊 Comparison Table")
        st.table({
            "Model": ["Baseline", "EACT", "RG-CLD"],
            "BLEU Score": [bleu_b, bleu_e, bleu_r],
            "EFC Score": [efc_b, efc_e, efc_r],
            "Hallucination Rate": [hall_b, hall_e, hall_r],
            "Semantic Similarity": [sem_b, sem_e, sem_r]
        })
