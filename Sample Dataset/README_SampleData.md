# 📂 Sample Dataset Documentation

This repository contains curated datasets designed for modeling and testing classification algorithms that predict **alteration types** based on geochemical and proxy variables.

---

## 📁 Dataset Structure

###  Traditional Sample Data

The **traditional datasets** are derived from the original, complete geochemical dataset `alldata.csv` (no missing values). The processing steps were as follows:

- A **10% random sample** was extracted from `alldata.csv`.
- This sample was split into:
  - `Trad_TrainDemo.csv` – Training set (70%)
  - `Trad_TestDemo.csv` – Testing set (30%)

These datasets contain the **original geochemical features** and serve as a baseline for evaluating model performance.

---

###  Simulated Sample Data

To test model generalization and performance on synthetic feature inputs, **simulated datasets** were created:

1. Starting from `alldata.csv` and `training.csv`, **proxy variables** were generated using custom MATLAB scripts.
2. These simulations produced:
   - `proxies_alldata.csv` – Full dataset with proxy features.
   - `proxies_training.csv` – Training portion with proxy features.
3. The test dataset was created by substracting the training portion from the full dataset.

Final processed files after decoding Alteration:

- `Simu_TrainDemo.csv` – Simulated training set
- `Simu_TestDemo.csv` – Simulated testing set

**Note: Values are in Gaussian scale, representing the distribution across multiple geostatistical realizations, so values may be negative.**

---

## 🔁 Alteration Code Conversion

In the MATLAB-generated simulated datasets, the `Alteration` class labels were initially **encoded as integers**:

| Code | Label |
|------|-------|
| 1    | AAA   |
| 2    | IAA   |
| 3    | PHY   |
| 4    | PRO   |
| 5    | PTS   |
| 6    | UAL   |

These were **decoded back to their original categorical labels** in the final `.csv` files to ensure consistency with the traditional datasets and compatibility with Python-based classification models.

---

> ✅ For optimal model training and evaluation, ensure consistent label encoding across both traditional and simulated datasets.

