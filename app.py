import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate

from utils.scoring import compute_bleu, compute_efc

st.set_page_config(
    page_title="MT Evaluation Dashboard",
    layout="wide"
)

# -------------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------------
st.markdown("""
    <h2 style="text-align:center; margin-bottom: 5px;">
        🌐 Premium Machine Translation Evaluation Dashboard
    </h2>
    <p style="text-align:center; color:#888; font-size:16px;">
        Analyze performance across Baseline, EACT, and RG-CLD models using advanced linguistic and factual metrics.
    </p>
    <hr>
""", unsafe_allow_html=True)

# =========================================================================
# USER INPUT
# =========================================================================
with st.container():
    st.subheader("🔤 Input Text")
    text = st.text_area("Enter text to evaluate:", height=120)

# =========================================================================
# BUTTON
# =========================================================================
if st.button("Run Evaluation"):
    if not text.strip():
        st.error("Please enter text first.")
    else:
        # ================================================================
        # INTERNAL MODEL RUNS (TRANSLATIONS NOT DISPLAYED)
        # ================================================================
        baseline_out = baseline_translate(text)
        eact_out = eact_translate(text)
        rgcld_out = rgcld_translate(text)

        # ------------------------------------------------------------------
        # METRIC CALCULATIONS
        # ------------------------------------------------------------------
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
        # TOP METRIC CARDS (DASHBOARD KPIs)
        # =========================================================================
        st.subheader("📊 Key Performance Indicators")

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown(f"""
                <div style="padding:20px; background:#f2f2f2; border-radius:10px;">
                <h3 style="margin:0;">Baseline</h3>
                <p>BLEU: <b>{bleu_b}</b></p>
                <p>EFC: <b>{efc_b}</b></p>
                </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
                <div style="padding:20px; background:#f2f2f2; border-radius:10px;">
                <h3 style="margin:0;">EACT Model</h3>
                <p>BLEU: <b>{bleu_e}</b></p>
                <p>EFC: <b>{efc_e}</b></p>
                </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
                <div style="padding:20px; background:#f2f2f2; border-radius:10px;">
                <h3 style="margin:0;">RG-CLD Model</h3>
                <p>BLEU: <b>{bleu_r}</b></p>
                <p>EFC: <b>{efc_r}</b></p>
                </div>
            """, unsafe_allow_html=True)

        # =========================================================================
        # TABS SECTION
        # =========================================================================
        tab1, tab2, tab3 = st.tabs(["📈 Primary Metrics", "📉 Advanced Metrics", "📊 Comparison Matrix"])

        # -------------------------------------------------------------------------
        # TAB 1 — PRIMARY METRICS
        # -------------------------------------------------------------------------
        with tab1:
            st.write("### 📈 BLEU & EFC Metrics (Per Model)")

            col1, col2, col3 = st.columns(3)

            # Baseline Graph
            with col1:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["BLEU", "EFC"], [bleu_b, efc_b])
                ax.set_ylim(0,1)
                ax.set_title("Baseline")
                st.pyplot(fig)

            # EACT Graph
            with col2:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["BLEU", "EFC"], [bleu_e, efc_e])
                ax.set_ylim(0,1)
                ax.set_title("EACT")
                st.pyplot(fig)

            # RG-CLD Graph
            with col3:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["BLEU", "EFC"], [bleu_r, efc_r])
                ax.set_ylim(0,1)
                ax.set_title("RG-CLD")
                st.pyplot(fig)

        # -------------------------------------------------------------------------
        # TAB 2 — ADVANCED METRICS
        # -------------------------------------------------------------------------
        with tab2:
            st.write("### 🧠 Hallucination & Semantic Similarity Analysis")

            colA, colB = st.columns(2)

            with colA:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["Baseline", "EACT", "RG-CLD"], [hall_b, hall_e, hall_r])
                ax.set_title("Hallucination Rate")
                ax.set_ylim(0, 1)
                st.pyplot(fig)

            with colB:
                fig, ax = plt.subplots(figsize=(3,3))
                ax.bar(["Baseline", "EACT", "RG-CLD"], [sem_b, sem_e, sem_r])
                ax.set_title("Semantic Similarity")
                ax.set_ylim(0, 1)
                st.pyplot(fig)

        # -------------------------------------------------------------------------
        # TAB 3 — COMPARISON MATRIX
        # -------------------------------------------------------------------------
        with tab3:
            st.write("### 📊 Score Comparison Table")

            st.table({
                "Model": ["Baseline", "EACT", "RG-CLD"],
                "BLEU": [bleu_b, bleu_e, bleu_r],
                "EFC": [efc_b, efc_e, efc_r],
                "Hallucination Rate": [hall_b, hall_e, hall_r],
                "Semantic Sim": [sem_b, sem_e, sem_r]
            })
