# MetaCost-XGB-Cost-Sensitive-Learning-with-Simulated-and-Traditional-Data
This repository contains Python codes for analyzing simulated and traditional datasets using the MetaCost algorithm and other utilities. The workflow involves generating predictions, aggregating results, and computing performance metrics.
## Folder Structure
```
.
├── XGBMetaCostSim.py
├── SimMode.py
├── SimModeMetrics.py
├── XGBMetaCostTrad.py
├── AccuracyFrequency.py
├── Bootstrapping.py
├── (training/testing datasets as specified in the code)
```

## Prerequisites
- Python >= 3.7
- Required Python packages:
  - numpy
  - pandas
  - scikit-learn
  - xgboost
  - scipy

Install dependencies using:
```bash
pip install numpy pandas scikit-learn xgboost scipy
```
## Demo Files Provided:
For testing purposes, demo dataset files named Simu_TrainDemo.csv , Simu_TestDemo.csv, Trad_TrainDemo.csv and Trad_TestDemo.csv have been included in the folder named Sample Dataset. These can be used to quickly run and verify the workflows.

## 1. Simulated Dataset Workflow

### Step 1: Generate Predictions
Run the following script to train the model on simulated data and generate predictions:
```bash
python XGBMetaCostSim.py
```
- Output: `MetaCostPredictions.csv`# MetaCost-XGB-Cost-Sensitive-Learning-with-Simulated-and-Traditional-Data

This repository contains Python scripts for analyzing simulated and traditional datasets using a MetaCost-based cost-sensitive learning approach with XGBoost. The workflow includes training, prediction, aggregation, and evaluation of model performance.

## Folder Structure
```
.
├── XGBMetaCostSim.py
├── SimMode.py
├── SimModeMetrics.py
├── XGBMetaCostTrad.py
├── AccuracyFrequency.py
├── Bootstrapping.py
├── (training/testing datasets as specified in the code)
```

## Prerequisites
- Python >= 3.7  
- Required Python packages:
  - numpy
  - pandas
  - scikit-learn
  - xgboost
  - scipy

Install dependencies using:
```bash
pip install numpy pandas scikit-learn xgboost scipy
```

## Datasets

All required training and testing datasets (`Simu_Train.csv`, `Simu_Test.csv`, `Trad_Train.csv`, `Trad_Test.csv`) are available for download from the **Zenodo repository** associated with this project. These files are necessary to run the workflows and are assumed to be placed in the appropriate directories as specified within each script.

---

## 1. Simulated Dataset Workflow

### Step 1: Generate Predictions
Run the following script to train a MetaCost-enhanced XGBoost model on simulated data:
```bash
python XGBMetaCostSim.py
```
- Input: Simu_Train.csv and Simu_Test.csv
- Output: `MetaCostPredictions.csv`

### Step 2: Compute Mode for Simulated Predictions
This aggregates predictions from multiple simulated samples by mode:
```bash
python SimMode.py
```
- Input: `MetaCostPredictions.csv`  
- Output: `MetaCost_PredictionsMode.csv` (includes a new `ModePred` column)

### Step 3: Compute Evaluation Metrics
Evaluate cost-sensitive metrics based on mode predictions:
```bash
python SimModeMetrics.py
```
- Input: `MetaCost_PredictionsMode.csv`  
- Output: Console report of performance metrics

---

## 2. Traditional Dataset Workflow

Run the MetaCost-enhanced model on traditional datasets:
```bash
python XGBMetaCostTrad.py
```
- Input: Trad_Train.csv and Trad_Test.csv   
- Output: Predictions and evaluation metrics printed or saved as specified

---

## 3. Accuracy and Frequency Analysis (Simulated)
Analyze prediction confidence based on class frequency:
```bash
python AccuracyFrequency.py
```
- Input: `MetaCost_PredictionsMode.csv`  
- Output: Console report of average accuracy by frequency class

---

## 4. Bootstrapping Analysis (Traditional)
Estimate prediction confidence using bootstrapped replicates:
```bash
python Bootstrapping.py
```
- Input: Trad_Train.csv and Trad_Test.csv  
- Output: Bootstrapped predictions with confidence scores

---

## Notes
- Ensure dataset files are downloaded from **Zenodo** and placed correctly.


