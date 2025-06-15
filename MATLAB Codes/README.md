# Geostatistical Proxy Generation for Regionalized Data

This repository contains MATLAB scripts for generating geostatistical proxies from an original dataset. The process expands the dataset by simulating 50 scenarios for each observation, which is especially useful for incorporating spatial uncertainty into machine learning workflows.

## Input Files

### `training.csv`
- **Description**: Original training dataset.
- **Format**: GSLIB-style header with 12 columns.
- **Columns**:
  1. `easting`
  2. `northing`
  3. `elevation`
  4. `alteration class` (categorical response variable) encoded as 1='AAA', 2='IAA', 3='PHY', 4='PRO', 5='PTS' and 6='UAL'
  5. 7 feature variables: `Cu`, `Au`, `Mo`, `As`, `Cp`, `Py`, `TS`

### `alldata.csv`
- **Description**: Combined training and testing dataset.

## How to Run

To generate the proxy datasets:

1. Open MATLAB.
2. Run the script `instructions.m`.

This script will automatically read the input files and generate simulated proxies for each row, using geostatistical techniques across 50 different realizations.

## Output Files

### `proxies_training.csv`
- **Description**: Simulated proxies for the training dataset.
- **Rows**: 50 times the original number of rows in `training.csv`.
- **Columns**: Same 12 columns as the input file.
- **Note**: Feature variables are now in **Gaussian scale**, so values may be negative.

### `proxies_alldata.csv`
- **Description**: Simulated proxies for the full dataset (training + testing).
- **Rows**: 50 times the original number of rows in `alldata.csv`.
- **Columns**: Same as input.
- **Note**: Values are in **Gaussian scale**, representing the distribution across multiple geostatistical realizations.

## Notes

- The Gaussian transformation enables use in machine learning pipelines that benefit from normalized, continuous input features.
- The 50 scenarios represent spatial uncertainty, helping build more robust predictive models.

## Requirements

- MATLAB (one has to previously install Matlab Runtime 2023b for Windows 64 bits)
- CSV input files in the correct format.

