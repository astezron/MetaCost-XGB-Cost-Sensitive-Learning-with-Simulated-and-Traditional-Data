# MATLAB-based geostatistical simulations and MetaCost-XGB: Cost-Sensitive Learning with Simulated and Traditional Data

This repository contains a complete workflow for applying the MetaCost algorithm to both simulated and traditional datasets(uploaded on Zenodo). The pipeline includes MATLAB codes for proxy generation and Python-based machine learning codes.

---

## Repository Structure

```bash
MATLAB-based geostatistical simulations and MetaCost-XGB-Cost-Sensitive-Learning-with-Simulated-and-Traditional-Data/
│
├── MATLAB Codes/              # MATLAB scripts for data simulation and proxy generation
│   └── README_Matlab.md       # Instructions for running the MATLAB portion
│
├── Python Codes/              # Python scripts for training, predictions, and metrics
│   └── README_Python.md       # Instructions for running Python-based MetaCost workflow
│
├── LICENSE                    # MIT License
└── README.md                  # Main description of the repository
```

---

## Workflow Overview

This project is built on a **two-phase workflow**:

### 1. **MATLAB Phase**
- Generates synthetic proxy datasets based on geostatistical simulation.
- Produces `proxies_alldata.csv`
- See [MATLAB Codes/README_Matlab.md](MATLAB%20Codes/README_Matlab.md) for detailed steps.
  

### 2. **Python Phase**
- Applies cost-sensitive XGBoost classification using the MetaCost algorithm.
- See [Python Codes/README_Python.md](Python%20Codes/README_Python.md) for detailed steps.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

