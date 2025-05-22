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

## 1. Simulated Dataset Workflow

### Step 1: Generate Predictions
Run the following script to train the model on simulated data and generate predictions:
```bash
python XGBMetaCostSim.py
```
- Output: `MetaCostPredictions.csv`

### Step 2: Compute Mode for Simulated Predictions
Run the following script to compute the mode for each set of 50 simulated datapoints:
```bash
python SimMode.py
```
- Input: `MetaCostPredictions.csv`
- Output: `MetaCost_PredictionsMode.csv` (with additional `ModePred` column)

### Step 3: Compute Metrics from Mode Predictions
Run the following script to evaluate metrics based on the mode predictions:
```bash
python SimModeMetrics.py
```
- Input: `MetaCost_PredictionsMode.csv`
- Output: Console metrics report 

## 2. Traditional Dataset Workflow

### Run MetaCost on Traditional Dataset
```bash
python XGBMetaCostTrad.py
```
- Input: Defined training/testing sets (specified in code)
- Output: Predictions and metrics as per code.

## 3. Accuracy and Frequency Analysis (Simulated Dataset)
To compute average frequency and accuracy per frequency class:
```bash
python AccuracyFrequency.py
```
- Input: `MetaCost_PredictionsMode.csv`
- Output: Frequency-Accuracy report.

## 4. Bootstrapping Analysis (Traditional Dataset)
To get the most frequent prediction and confidence score per test sample using bootstrapping:
```bash
python Bootstrapping.py
```
- Input: Defined training/testing sets (specified in code)
- Output: Bootstrapped predictions and confidence scores.

## Notes:
- Ensure the training and testing datasets are correctly mentioned in each script. These are hardcoded and need no additional argument passing.

