import streamlit as st
import matplotlib.pyplot as plt

from models.baseline_model import baseline_translate
from models.eact_model import eact_translate
from models.rgcld_model import rgcld_translate

from utils.scoring import compute_bleu, compute_efc

st.set_page_config(page_title="Reliable Machine Translation", layout="wide")

st.title("🌐 Reliable Machine Translation Framework")
st.write("Enter text below to generate translations and compare model accuracy.")

text = st.text_area("Enter text for translation:", height=150)

if st.button("Translate"):
    if not text.strip():
        st.error("Please type something first.")
    else:
        # Generate model outputs
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

        # Compute accuracy scores
        bleu_scores = [
            compute_bleu(text, baseline_out),
            compute_bleu(text, eact_out),
            compute_bleu(text, rgcld_out)
        ]

        efc_scores = [
            compute_efc(text, baseline_out),
            compute_efc(text, eact_out),
            compute_efc(text, rgcld_out)
        ]

        labels = ["Baseline", "EACT", "RG-CLD"]

        st.subheader("Model Accuracy Comparison")

        # Plot scores
        fig, ax = plt.subplots()
        ax.plot(labels, bleu_scores, marker='o', label="BLEU Score")
        ax.plot(labels, efc_scores, marker='o', label="EFC Score")
        ax.set_title("Model Performance Overview")
        ax.set_xlabel("Models")
        ax.set_ylabel("Scores")
        ax.legend()

        st.pyplot(fig)
