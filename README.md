
# ⚡ Neural Machine Translation Insights & Evaluation Platform

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-%3E%3D3.8-green.svg)
![Status](https://img.shields.io/badge/build-stable-success)

A lightweight, developer-friendly platform for benchmarking, debugging, and interpreting Neural Machine Translation (NMT) models.  
Built for engineers who want **clarity in outputs**, **control over metrics**, and **speed in experimentation**.

---

## 🧠 What This Platform Delivers

- ⚙️ **Side-by-side comparison** of multiple NMT models  
- 📊 **BLEU** + **Entity-Focused Correctness (EFC)** for semantic evaluation  
- 🖥️ **Streamlit-powered UI** for rapid iteration & testing  
- 🔍 Visual debugging tools to understand model failures  
- 🔧 Modular codebase that fits naturally into research or production R&D

This repo is crafted to give you **real insights**, not just output strings.

---

## 🚀 Core Features

### 🏗 Model Variants
- Transformer (Baseline)
- Entity-Aware Contrastive Fine-Tuning (EACT)
- Retrieval-Guided Constrained Lattice Decoding (RG-CLD)

### 📐 Metrics
- BLEU Score
- Entity-Focused Correctness (EFC)  
  *(Because token-level metrics alone don’t tell the whole story.)*

### 🖥 Developer UI
- Real-time inference  
- Multi-model output comparison  
- Entity-level alignment visualizations  
- Metric plots & debugging hooks  

---

## ⚡ Quickstart

### Clone
```bash
git clone https://github.com/caffe-cappuccino/Neural_Machine_Translation_Insights_Evaluation_Platform.git
cd Neural_Machine_Translation_Insights_Evaluation_Platform
````

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

---

## 📂 Directory Layout

```
.
├── models/            # Checkpoints, retrieval files, model weights
├── utils/             # Metrics, preprocessors, helpers
├── app.py             # Streamlit frontend
├── requirements.txt
└── README.md
```

Everything is modular. Everything is hackable.

---

## 🧩 Workflow

1. Input text or upload a dataset
2. Generate translations across selected models
3. Compare metrics (BLEU + EFC)
4. Explore visual breakdowns
5. Debug entity mismatches, drift, hallucinations
6. Iterate fast. Deploy faster.

---

## 🔧 Extend Like a Pro

Adding a new model?

```bash
# drop model files → /models
# implement wrapper → /utils
# register in UI → app.py
```

Adding a new metric?

```bash
# write metric function → /utils
# add to evaluation pipeline
# visualize in Streamlit
```

This repo is designed with **clean separations**, **functional modules**, and **plug-and-play architecture**.

---

## 📖 References

* *Attention Is All You Need*, Vaswani et al. (2017)
* Research on Entity-Aware MT Fine-Tuning
* Retrieval-guided decoding approaches

---

## 🤝 Contributions

Pull requests, issue reports, and model extensions are always welcome.
If you break something, improve something, or optimize something — ship it 🚀

---

## 📝 License

MIT License — free to build, break, improve, and ship.

---

**Built for engineers who care about clarity, metrics, and control.**
Happy hacking ⚡

