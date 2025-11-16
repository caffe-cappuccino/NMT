import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate

from utils.scoring import compute_bleu, compute_efc

# -------------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="MT Evaluation Dashboard",
    layout="wide",
)

# -------------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------------
st.markdown("""
    <h1 style="text-align:center; margin-bottom: 5px;">
        🌐 Ultra-Premium MT Evaluation Dashboard
    </h1>
    <p style="text-align:center; color:#bbb; font-size:17px;">
        Benchmark reliability, factual consistency, and semantic integrity across Baseline, EACT, and RG-CLD models.
    </p>
    <hr>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# INPUT SECTION
# -------------------------------------------------------------------------
st.subheader("📝 Enter Text")
text = st.text_area("Text to Evaluate:", height=130)

# -------------------------------------------------------------------------
# PROCESS BUTTON
# -------------------------------------------------------------------------
if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Please enter text.")
    else:

        # =========================================================================
        # MODEL OUTPUTS (Hidden from UI)
        # =========================================================================
        baseline_out = baseline_translate(text)
        eact_out = eact_translate(text)
        rgcld_out = rgcld_translate(text)

        # =========================================================================
        # METRIC FUNCTION
        # =========================================================================
        def get_metrics(output):
            bleu = compute_bleu(text, output)
            efc = compute_efc(text, output)
            halluc = round(1 - efc, 3)
            sem_sim = round((bleu + efc) / 2, 3)
            return bleu, efc, halluc, sem_sim

        bleu_b, efc_b, hall_b, sem_b = get_metrics(baseline_out)
        bleu_e, efc_e, hall_e, sem_e = get_metrics(eact_out)
        bleu_r, efc_r, hall_r, sem_r = get_metrics(rgcld_out)

        # =========================================================================
        # ULTRA-PREMIUM GLASS KPI CARDS
        # =========================================================================
        st.subheader("💎 Ultra-Premium KPI Overview")

        card_css = """
            <style>
            .kpi-card {
                background: rgba(255, 255, 255, 0.15);
                backdrop-filter: blur(15px);
                -webkit-backdrop-filter: blur(15px);
                border-radius: 18px;
                padding: 25px;
                border: 1px solid rgba(255,255,255,0.25);
                box-shadow: 0 8px 25px rgba(0,0,0,0.25);
                color: #fff;
                margin-bottom: 20px;
            }
            .kpi-title {
                font-size: 22px;
                font-weight: 700;
                text-shadow: 0 0 6px rgba(0,0,0,0.4);
            }
            .kpi-value {
                font-size: 18px;
                margin-top: 8px;
                font-weight: 600;
            }
            </style>
        """

        st.markdown(card_css, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                <div class="kpi-card" style="border-left: 5px solid #4facfe;">
                    <div class="kpi-title">Baseline Model</div>
                    <div class="kpi-value">BLEU Score: <b>{bleu_b}</b></div>
                    <div class="kpi-value">EFC Score: <b>{efc_b}</b></div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="kpi-card" style="border-left: 5px solid #43e97b;">
                    <div class="kpi-title">EACT Model</div>
                    <div class="kpi-value">BLEU Score: <b>{bleu_e}</b></div>
                    <div class="kpi-value">EFC Score: <b>{efc_e}</b></div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class="kpi-card" style="border-left: 5px solid #fa709a;">
                    <div class="kpi-title">RG-CLD Model</div>
                    <div class="kpi-value">BLEU Score: <b>{bleu_r}</b></div>
                    <div class="kpi-value">EFC Score: <b>{efc_r}</b></div>
                </div>
            """, unsafe_allow_html=True)

        # =========================================================================
        # TABS FOR METRICS
        # =========================================================================
        tab1, tab2, tab3 = st.tabs(["📈 Primary Metrics", "🧠 Advanced Metrics", "📊 Comparison Table"])

        # -------------------------------------------------------------------------
        # TAB 1 — PRIMARY METRICS
        # -------------------------------------------------------------------------
        with tab1:
            st.write("### 📈 BLEU & EFC Metric Visualization")

            colA, colB, colC = st.columns(3)

            with colA:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["BLEU", "EFC"], [bleu_b, efc_b], color=["#4facfe", "#00f2fe"])
                ax.set_ylim(0, 1)
                ax.set_title("Baseline")
                st.pyplot(fig)

            with colB:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["BLEU", "EFC"], [bleu_e, efc_e], color=["#43e97b", "#38f9d7"])
                ax.set_ylim(0, 1)
                ax.set_title("EACT")
                st.pyplot(fig)

            with colC:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["BLEU", "EFC"], [bleu_r, efc_r], color=["#fa709a", "#fee140"])
                ax.set_ylim(0, 1)
                ax.set_title("RG-CLD")
                st.pyplot(fig)

        # -------------------------------------------------------------------------
        # TAB 2 — ADVANCED METRICS
        # -------------------------------------------------------------------------
        with tab2:
            st.write("### 🧠 Hallucination & Semantic Similarity")

            colX, colY = st.columns(2)

            with colX:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["Baseline", "EACT", "RG-CLD"], [hall_b, hall_e, hall_r], color="#ff4e50")
                ax.set_ylim(0, 1)
                ax.set_title("Hallucination Rate (Lower = Better)")
                st.pyplot(fig)

            with colY:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["Baseline", "EACT", "RG-CLD"], [sem_b, sem_e, sem_r], color="#30cfd0")
                ax.set_ylim(0, 1)
                ax.set_title("Semantic Similarity (Higher = Better)")
                st.pyplot(fig)

        # -------------------------------------------------------------------------
        # TAB 3 — COMPARISON TABLE
        # -------------------------------------------------------------------------
        with tab3:
            st.write("### 📊 Model Comparison Matrix")
            st.table({
                "Model": ["Baseline", "EACT", "RG-CLD"],
                "BLEU Score": [bleu_b, bleu_e, bleu_r],
                "EFC Score": [efc_b, efc_e, efc_r],
                "Hallucination Rate": [hall_b, hall_e, hall_r],
                "Semantic Similarity": [sem_b, sem_e, sem_r]
            })
