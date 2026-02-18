# Epigenetic Age Prediction with LightGBM

High-dimensional regression on ~37,000 CpG methylation features to predict relative biological age across mammalian species.

Dataset:  
Haghani et al., *DNA methylation networks underlying mammalian traits*, Science (2023)  
https://www.science.org/doi/10.1126/science.abq5693

---

## 🚀 Problem

Predict relative biological age from DNA methylation profiles.

Challenges:

- ~37,000 CpG features (extreme high dimensionality)
- Strong multicollinearity
- Non-linear feature–target relationships
- Heterogeneous samples across species and tissues

This project evaluates whether gradient boosting (LightGBM) can effectively model complex aging signals in high-dimensional biological data.

---

## 📊 Dataset

- ~12,000 mammalian samples  
- ~37,000 CpG methylation features per sample  
- Target: **Relative age** (normalized by species lifespan)

Methylation values are beta values representing CpG methylation levels.

---

## 🔄 Target Transformation

Following the original publication, the model was trained on:

**Double negative log-transformed relative age**

This stabilizes variance and reduces skew.

Predictions were inverse-transformed to obtain interpretable relative age estimates.

---

## 🧠 Model Choice: LightGBM

LightGBM was selected because it:

- Performs strongly on tabular data
- Handles high-dimensional feature spaces efficiently
- Is robust to multicollinearity
- Performs implicit feature selection via split gain

Model type: Gradient Boosted Decision Trees (regression)

---

## ⚙️ Training Pipeline

1. Load and preprocess methylation matrix  
2. Train/test split  
3. Target transformation  
4. LightGBM model training  
5. Prediction  
6. Inverse transformation  
7. Evaluation  

Reproducible and modular workflow.

---

## 📈 Evaluation Metrics

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- Pearson correlation coefficient  

These measure absolute error and alignment with biological age.

---

## 🔬 Engineering Highlights

- Trained on ~37k-dimensional feature space  
- Modeled complex nonlinear aging patterns  
- Applied variance-stabilizing transformations  
- Compared against linear (Elastic Net) and neural network baselines  
- Designed reproducible ML workflow  

---

## 🛠 Tech Stack

- Python  
- LightGBM  
- scikit-learn  
- pandas  
- numpy  

---

## ▶ Reproducibility

```bash
pip install -r requirements.txt
python train_lightgbm.py

