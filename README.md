# MATLAB-based geostatistical simulations and MetaCost-XGB: Cost-Sensitive Learning with Simulated and Traditional Data

This repository contains a complete workflow for applying the MetaCost algorithm to both simulated and traditional datasets. The pipeline includes MATLAB-based geostatistical simulations, Python-based machine learning, and GitHub Actions automation.

---

## Repository Structure

```bash
MATLAB-based geostatistical simulations and MetaCost-XGB-Cost-Sensitive-Learning-with-Simulated-and-Traditional-Data/
│
├── .github/workflows/         # GitHub Actions workflow files
│   └── run-simulated.yml      # Defines CI for running the pipeline on demo datasets
│
├── Matlab Codes/              # MATLAB scripts for data simulation and proxy generation
│   └── README_Matlab.md       # Instructions for running the MATLAB portion
│
├── Python Codes/              # Python scripts for training, predictions, and metrics
│   └── README_Python.md       # Instructions for running Python-based MetaCost workflow
│
├── Sample Dataset/            # Contains demo input and output CSV files
│   ├── Simu_TrainDemo.csv     # Training dataset for Simulations demo
│   ├── Simu_TestDemo.csv      # Testing dataset for Simulations demo
│   ├── Trad_TrainDemo.csv     # Training dataset for Traditional demo
│   └── Trad_TestDemo.csv      # Testing dataset for Traditional demo
│
├── LICENSE                    # MIT License
└── README.md                  # Main description of the repository
```

---

## Workflow Overview

This project is built on a **two-phase workflow**:

### 1. **MATLAB Phase**
- Generates synthetic proxy datasets based on geostatistical simulation.
- Produces `proxies_training.csv` and `proxies_alldata.csv`.
- See [Matlab Codes/README_Matlab.md](Matlab%20Codes/README_Matlab.md) for detailed steps.
  

### 2. **Python Phase**
- Applies cost-sensitive XGBoost classification using the MetaCost algorithm.
- See [Python Codes/README_Python.md](Python%20Codes/README_Python.md) for detailed steps.

---

## Demo & Automation

A demo dataset – 10% sample of the original dataset – is provided in `Sample Dataset/`:
  - `Simu_TrainDemo.csv` and `Simu_TestDemo.csv` for the **Simulated Dataset**
  - `Trad_TrainDemo.csv` and `Trad_TestDemo.csv` for the **Traditional Dataset**
- This demo is designed to test the **MetaCost XGBoost algorithm** on both traditional and simulated datasets.
- The pipeline runs end-to-end: training, predicting, aggregating results, and evaluating metrics, simulating the full workflow on smaller data.
- GitHub Actions automation runs this demo pipeline every time the key files are updated.
- Workflow YAML: [.github/workflows/run-simulated.yml](.github/workflows/run-simulated.yml)

**Expected run time on a normal desktop:**

- Traditional Demo: ~1 minute

- Simulated Demo: ~20 minutes

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

