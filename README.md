

# Neural Machine Translation Insights & Evaluation Platform

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-%3E%3D3.8-green.svg)

An end-to-end interactive platform for evaluating, comparing, and visualizing Neural Machine Translation (NMT) models.  
Designed to streamline experimentation, boost interpretability, and deliver deep insights into translation quality.

---

## 🚀 Overview

This platform empowers researchers and engineers to:

- Benchmark a **baseline Transformer** against advanced NMT variants.
- Evaluate translations using **BLEU** and **Entity-Focused Correctness (EFC)**.
- Interactively visualize outputs, metrics, and semantic alignment.
- Test, compare, and debug real-world translation behaviours quickly through a clean Streamlit UI.

Perfect for rapid experimentation, research projects, and production-ready evaluation workflows.

---

## 🔍 Features

- **Model Variants Included**
  - Baseline Transformer
  - Entity-Aware Contrastive Fine-Tuning (EACT)
  - Retrieval-Guided Constrained Lattice Decoding (RG-CLD)

- **Metrics**
  - BLEU Score
  - Entity-Focused Correctness (EFC)

- **Streamlit UI**
  - Enter or upload source text
  - Generate model outputs
  - Compare side-by-side translations
  - Visual metric breakdowns, graphs, and insights

- **Plug-and-Play Architecture**
  - Add new models easily
  - Add new evaluation metrics
  - Extend visualizations or integrate into pipelines

---

## 🧰 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/caffe-cappuccino/Neural_Machine_Translation_Insights_Evaluation_Platform.git
cd Neural_Machine_Translation_Insights_Evaluation_Platform
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Neural_Machine_Translation_Insights_Evaluation_Platform/
│
├── models/            # Pre-trained models & retrieval assets
├── utils/             # Metrics, preprocessing, helper utilities
├── app.py             # Main Streamlit application
├── requirements.txt   # Python dependencies
└── README.md
```

---

## ✨ How to Use

1. Run the app using Streamlit.
2. Enter text or upload a dataset file.
3. Generate translations using each model.
4. Compare BLEU & EFC across outputs.
5. Explore visual insights – entity matches, error segments, metric scores, plots.
6. Export logs or integrate into your workflow.

---

## 🔧 Extending the Platform

You can enhance the system by:

* Adding new NMT architectures (e.g., mBART, Marian, LLaMA-based models).
* Integrating metrics like COMET, BLEURT, Token-Level F1, or custom scoring.
* Building advanced dashboards (A/B comparison, error tagging, drift detection).
* Automating evaluations for CI/CD pipelines.

---

## 📖 References

* Vaswani et al. *“Attention Is All You Need”* (2017)
* Research on Entity-Aware/Contrastive NMT Fine-Tuning
* Retrieval-Guided Decoding Frameworks

---

## 🤝 Contributing

Contributions are warmly welcomed!
Feel free to open issues, suggest features, or submit pull requests.

---

## 📝 License

This project is released under the **MIT License**.

---

Thanks for exploring this platform!
Let's elevate machine translation evaluation to the next level 🚀✨

