import streamlit as st
import matplotlib.pyplot as plt

from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate

from utils.scoring import compute_bleu, compute_efc

st.set_page_config(page_title="Reliable Machine Translation", layout="wide")

st.title("🌐 Reliable Machine Translation Framework")
st.write("Enter text below to generate translations and visualize model evaluation metrics.")

text = st.text_area("Enter text for translation:", height=150)

if st.button("Translate"):
    if not text.strip():
        st.error("Please enter text first.")
    else:

        # -------- MODEL OUTPUTS --------
        baseline_out = baseline_translate(text)
        eact_out = eact_translate(text)
        rgcld_out = rgcld_translate(text)

        st.subheader("Translations")
        st.write("**Baseline Transformer:**")
        st.success(baseline_out)

        st.write("**EACT Fine-Tuned Model:**")
        st.success(eact_out)

        st.write("**RG-CLD Retrieval-Guided Model:**")
        st.success(rgcld_out)

        # -------- METRICS --------
        bleu_baseline = compute_bleu(text, baseline_out)
        efc_baseline = compute_efc(text, baseline_out)

        bleu_eact = compute_bleu(text, eact_out)
        efc_eact = compute_efc(text, eact_out)

        bleu_rgcld = compute_bleu(text, rgcld_out)
        efc_rgcld = compute_efc(text, rgcld_out)

        # -------- SEPARATE GRAPHS --------
        st.subheader("Model Evaluation Graphs")

        # Baseline Graph
        st.write("### 📘 Baseline Transformer Metrics")
        fig1, ax1 = plt.subplots()
        ax1.bar(["BLEU", "EFC"], [bleu_baseline, efc_baseline])
        ax1.set_title("Baseline Model Performance")
        ax1.set_ylim(0, 1)
        st.pyplot(fig1)

        # EACT Graph
        st.write("### 🟦 EACT Fine-Tuned Model Metrics")
        fig2, ax2 = plt.subplots()
        ax2.bar(["BLEU", "EFC"], [bleu_eact, efc_eact])
        ax2.set_title("EACT Model Performance")
        ax2.set_ylim(0, 1)
        st.pyplot(fig2)

        # RG-CLD Graph
        st.write("### 🟩 RG-CLD Retrieval-Guided Model Metrics")
        fig3, ax3 = plt.subplots()
        ax3.bar(["BLEU", "EFC"], [bleu_rgcld, efc_rgcld])
        ax3.set_title("RG-CLD Model Performance")
        ax3.set_ylim(0, 1)
        st.pyplot(fig3)
