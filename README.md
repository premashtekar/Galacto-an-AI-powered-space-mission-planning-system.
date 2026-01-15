# 🌌 GALACTO – AI-Powered Space Mission Planning System

GALACTO is my **first solo end-to-end Machine Learning project**, built to explore how AI can assist in **space mission planning and decision-making**.

This project predicts key mission parameters and benchmarks them against **historical space missions**, making predictions more interpretable and realistic.

---

## 🚀 What GALACTO Does

Given mission parameters, GALACTO predicts:

- 🛰️ **Mission Type** (Exploration / Colonization / Mining)
- 💰 **Estimated Mission Cost** (in Billion USD)
- 📈 **Mission Success Probability (%)**

Additionally, it visualizes:
- 📊 **Historical cost comparison** for similar missions (Mars / Moon / etc.)

This helps answer:
> *“Is this mission realistic compared to past missions?”*

---

## 📊 Datasets Used

- Historical space mission data  
- Mission targets (Mars, Moon, Titan, etc.)
- Launch vehicles
- Mission cost and success percentages  

All categorical data is encoded properly before training.

---

## 🧠 Tech Stack

- **Python**
- **Scikit-learn**
- **Pandas / NumPy**
- **Streamlit** (interactive dashboard)
- **Matplotlib** (visualizations)

---

## 🧩 ML Models Used

- Random Forest **Classifier** → Mission Type
- Random Forest **Regressor** → Mission Cost
- Random Forest **Regressor** → Mission Success

---

## 🖥️ How to Run Locally

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
streamlit run app.py
