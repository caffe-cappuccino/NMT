import streamlit as st
import matplotlib.pyplot as plt

from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate

from utils.scoring import compute_bleu, compute_efc

st.set_page_config(page_title="Reliable Machine Translation Dashboard", layout="wide")

st.title("🌐 Reliable Machine Translation Framework")
st.write("Compare model outputs and evaluation metrics for Baseline, EACT, and RG-CLD models.")

# -------------------------------------------------------------------------
# USER INPUT
# -------------------------------------------------------------------------
text = st.text_area("Enter text for translation:", height=150)

# -------------------------------------------------------------------------
# ON BUTTON CLICK
# -------------------------------------------------------------------------
if st.button("Translate"):
    if not text.strip():
        st.error("Please enter text first.")
    else:

        # =====================================================================
        # 1. MODEL TRANSLATIONS (THIS PART WAS NOT SHOWING EARLIER)
        # =====================================================================
        baseline_out = baseline_translate(text)
        eact_out = eact_translate(text)
        rgcld_out = rgcld_translate(text)

        st.subheader("Translations")

        colT1, colT2, colT3 = st.columns(3)

        with colT1:
            st.markdown("### Baseline Transformer")
            st.success(baseline_out)

        with colT2:
            st.markdown("### EACT Fine-Tuned")
            st.success(eact_out)

        with colT3:
            st.markdown("### RG-CLD Retrieval-Guided")
            st.success(rgcld_out)

        # =====================================================================
        # 2. METRIC CALCULATIONS
        # =====================================================================

        # Baseline
        bleu_baseline = compute_bleu(text, baseline_out)
        efc_baseline = compute_efc(text, baseline_out)
        halluc_baseline = round(1 - efc_baseline, 3)
        semantic_baseline = round((bleu_baseline + efc_baseline) / 2, 3)

        # EACT
        bleu_eact = compute_bleu(text, eact_out)
        efc_eact = compute_efc(text, eact_out)
        halluc_eact = round(1 - efc_eact, 3)
        semantic_eact = round((bleu_eact + efc_eact) / 2, 3)

        # RG-CLD
        bleu_rgcld = compute_bleu(text, rgcld_out)
        efc_rgcld = compute_efc(text, rgcld_out)
        halluc_rgcld = round(1 - efc_rgcld, 3)
        semantic_rgcld = round((bleu_rgcld + efc_rgcld) / 2, 3)

        # =====================================================================
        # 3. SIDE-BY-SIDE MODEL GRAPHS (BLEU + EFC)
        # =====================================================================
        st.subheader("Evaluation Metrics (Per Model)")

        col1, col2, col3 = st.columns(3)

        # ---- Baseline Chart ----
        with col1:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            ax1.bar(["BLEU", "EFC"], [bleu_baseline, efc_baseline])
            ax1.set_title("Baseline")
            ax1.set_ylim(0, 1)
            st.pyplot(fig1)

        # ---- EACT Chart ----
        with col2:
            fig2, ax2 = plt.subplots(figsize=(3, 3))
            ax2.bar(["BLEU", "EFC"], [bleu_eact, efc_eact])
            ax2.set_title("EACT")
            ax2.set_ylim(0, 1)
            st.pyplot(fig2)

        # ---- RG-CLD Chart ----
        with col3:
            fig3, ax3 = plt.subplots(figsize=(3, 3))
            ax3.bar(["BLEU", "EFC"], [bleu_rgcld, efc_rgcld])
            ax3.set_title("RG-CLD")
            ax3.set_ylim(0, 1)
            st.pyplot(fig3)


        # =====================================================================
        # 4. ADVANCED METRICS (HALLUCINATION + SEMANTIC SIMILARITY)
        # =====================================================================
        st.subheader("Advanced Reliability Metrics")

        colA, colB = st.columns(2)

        # ---- Hallucination Rate Chart ----
        with colA:
            fig4, ax4 = plt.subplots(figsize=(3, 3))
            ax4.bar(["Baseline", "EACT", "RG-CLD"],
                    [halluc_baseline, halluc_eact, halluc_rgcld])
            ax4.set_title("Hallucination Rate (Lower is Better)")
            ax4.set_ylim(0, 1)
            st.pyplot(fig4)

        # ---- Semantic Similarity Chart ----
        with colB:
            fig5, ax5 = plt.subplots(figsize=(3, 3))
            ax5.bar(["Baseline", "EACT", "RG-CLD"],
                    [semantic_baseline, semantic_eact, semantic_rgcld])
            ax5.set_title("Semantic Similarity (Higher is Better)")
            ax5.set_ylim(0, 1)
            st.pyplot(fig5)
