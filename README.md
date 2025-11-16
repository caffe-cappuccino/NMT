# Reliable Neural Machine Translation – Streamlit App

This project demonstrates a hybrid NMT framework comparing:
- Baseline Transformer
- Entity-Aware Contrastive Fine-Tuning (EACT)
- Retrieval-Guided Constrained Lattice Decoding (RG-CLD)

The application:
✔ Takes user input  
✔ Generates translations  
✔ Computes BLEU + EFC scores  
✔ Visualizes accuracy using graphs  

### Deployment
The app is built for Streamlit Cloud.  
Just upload this folder to GitHub and connect it to Streamlit Cloud.

### Run Locally
pip install -r requirements.txt  
streamlit run app.py
