# Sample Dataset Information

This repository contains sample datasets used for modeling and testing classification algorithms for alteration type prediction based on geochemical and simulated proxy data.

## 📁 Dataset Overview

### Traditional Sample Data
The **traditional sample data** was derived from the original isotopic dataset `alldata.csv` (which contains no missing values). From this dataset:

- A 10% random sample was extracted.
- This sample was then split into:
  - `Trad_TrainDemo.csv` – Training dataset
  - `Trad_TestDemo.csv` – Testing dataset

These files represent the "traditional" form of the data, using original geochemical features.

### Simulated Sample Data
To enhance the dataset and test model performance on generated proxy features, **simulated data** was created:

1. Starting from `alldata.csv`, simulations were performed using custom MATLAB codes.
2. These simulations produced a new dataset: `proxies_alldata.csv`, containing proxy variables for geochemical indicators.
3. This was again split into:
   - `Simu_TrainDemo.csv` – Training dataset with proxies
   - `Simu_TestDemo.csv` – Testing dataset with proxies

## 🧠 Alteration Code Transformation

In the MATLAB-generated files, the `Alteration` field was **initially encoded as integers**:

| Code | Class |
|------|--------|
| 1    | AAA    |
| 2    | IAA    |
| 3    | PHY    |
| 4    | PRO    |
| 5    | PTS    |
| 6    | UAL    |

Before using these files in Python-based machine learning pipelines, the integer codes were **decoded back to their original categorical labels** to ensure consistency with the traditional dataset and proper label interpretation in classification tasks.

---

**Note:** All datasets are in `.csv` format and are stored locally for offline experimentation and analysis.
