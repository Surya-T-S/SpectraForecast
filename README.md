# Stock Price Forecasting - STFT Spectrogram + CNN

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.13-blue.svg" alt="python" />
    <img src="https://img.shields.io/badge/Framework-PyTorch-red.svg" alt="pytorch" />
    <img src="https://img.shields.io/badge/Domain-Financial%20Time%20Series-green.svg" alt="domain" />
    <img src="https://img.shields.io/badge/Approach-STFT%20%2B%20CNN-orange.svg" alt="approach" />
</p>

<p align="center">
    Time-frequency deep learning pipeline for stock forecasting using multi-channel spectrograms.
</p>

## Highlights
- End-to-end ML workflow: collection -> features -> spectrograms -> training -> evaluation -> ablation
- Time-frequency representation of financial signals using STFT
- CNN regressor with reproducible experiments and saved artifacts
- Comparative evaluation and ablation analysis with published figures

## Author
- Name: Surya T S
- University ID: TCR24CS069

## Overview
This repository presents an end-to-end stock forecasting system that converts financial time series into STFT spectrograms and predicts future close price using a CNN regressor.

The pipeline uses:
- Companies: RELIANCE.NS, TCS.NS, INFY.NS
- Context channels: Sensex (^BSESN), USD/INR (INR=X)
- Time range: 2015-01-01 to 2024-12-31

Core idea:
- Build a multivariate signal from market + technical indicators
- Convert each rolling window into a multi-channel spectrogram tensor
- Train CNN for regression on future raw close prices

## Tech Stack
- Python, NumPy, pandas, SciPy
- PyTorch for deep learning
- scikit-learn for preprocessing and metrics
- matplotlib for visualization
- yfinance for market data ingestion

## Quick Visuals

| CNN Architecture | Ablation Summary |
|---|---|
| ![CNN Architecture](./figures/cnn_architecture.png) | ![Ablation](./figures/ablation.png) |

| Metrics Comparison | Time-Series Snapshot |
|---|---|
| ![Metrics Comparison](./figures/metrics_comparison.png) | ![Time Series](./figures/time_series.png) |

## All Visual Outputs
All generated PNG files are available in `figures/` and visible on GitHub.

- [ablation.png](./figures/ablation.png)
- [cnn_architecture.png](./figures/cnn_architecture.png)
- [fft_spectrum.png](./figures/fft_spectrum.png)
- [metrics_comparison.png](./figures/metrics_comparison.png)
- [predictions_INFY.NS.png](./figures/predictions_INFY.NS.png)
- [predictions_RELIANCE.NS.png](./figures/predictions_RELIANCE.NS.png)
- [predictions_TCS.NS.png](./figures/predictions_TCS.NS.png)
- [spectrograms.png](./figures/spectrograms.png)
- [time_series.png](./figures/time_series.png)
- [training_curve_INFY.NS.png](./figures/training_curve_INFY.NS.png)
- [training_curve_RELIANCE.NS.png](./figures/training_curve_RELIANCE.NS.png)
- [training_curve_TCS.NS.png](./figures/training_curve_TCS.NS.png)

## Data Snapshot
- Raw downloads: `data/raw/`
- Engineered features and splits: `data/processed/`
- Spectrogram tensors: `data/spectrograms/`
- Trained checkpoints: `models/`
- Evaluation and experiment visuals: `figures/`

For complete implementation details, see the scripts in `src/`.

## Project Structure
```text
stock-forecasting/
├── config.yaml                          # Central experiment configuration and hyperparameters
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
├── .gitignore                           # Git exclusions for data/artifacts
├── data/
│   ├── raw/
│   │   ├── RELIANCE.NS.csv              # Raw close series
│   │   ├── TCS.NS.csv                   # Raw close series
│   │   ├── INFY.NS.csv                  # Raw close series
│   │   ├── ^BSESN.csv                   # Raw index close series
│   │   ├── INR=X.csv                    # Raw forex close series
│   │   └── merged_raw.csv               # Aligned merged close dataframe
│   ├── processed/
│   │   ├── {company}_features.csv       # Engineered feature matrix per company
│   │   ├── {company}_train.csv          # Normalized train split
│   │   ├── {company}_val.csv            # Normalized validation split
│   │   ├── {company}_test.csv           # Normalized test split
│   │   ├── {company}_scaler.pkl         # MinMax scaler fitted on train split
│   │   ├── metrics.csv                  # Evaluation metrics across companies
│   │   └── ablation_results.csv         # Structured ablation outputs
│   └── spectrograms/
│       ├── {company}_X_train.npy        # Train spectrogram tensors
│       ├── {company}_y_train.npy        # Train targets (raw close)
│       ├── {company}_X_val.npy          # Validation spectrogram tensors
│       ├── {company}_y_val.npy          # Validation targets
│       ├── {company}_X_test.npy         # Test spectrogram tensors
│       └── {company}_y_test.npy         # Test targets
├── figures/
│   ├── time_series.png                  # Raw close with split boundaries
│   ├── fft_spectrum.png                 # FFT magnitude plots
│   ├── spectrograms.png                 # Channel-0 sample spectrograms
│   ├── training_curve_{company}.png     # Training and validation loss curves
│   ├── predictions_{company}.png        # Evaluation visualization panels
│   ├── metrics_comparison.png           # RMSE and MAPE comparison chart
│   ├── ablation.png                     # Ablation visualization summary
│   └── cnn_architecture.png             # CNN block diagram
├── models/
│   └── {company}_best.pth               # Best checkpoint per company
├── notebooks/                           # Optional exploratory notebooks
└── src/
    ├── __init__.py
    ├── 01_data_collection.py            # Download and merge raw market data
    ├── 02_feature_engineering.py        # Feature generation, scaling, split exports
    ├── 03_spectrogram_generator.py      # STFT spectrogram sample creation
    ├── 04_dataset.py                    # PyTorch Dataset and DataLoader helpers
    ├── 05_model.py                      # SpectrogramCNN architecture definition
    ├── 06_train.py                      # Training loop, scheduler, early stopping
    ├── 07_evaluate.py                   # Test inference, metrics, and plots
    └── 08_ablation.py                   # Window/horizon/feature-set ablation study
```

## Methodology

### Signal Representation
Each stock is represented as a 7-channel financial signal $X(t)$:
1. close
2. log_return
3. volatility
4. RSI
5. MACD
6. sensex_close
7. usd_inr

These channels are date-aligned and processed with sliding windows.

### STFT Spectrogram Generation
For every channel and window:

$$
S(t,f) = |\mathrm{STFT}(t,f)|^2
$$

Configured defaults:
- Window length $L=32$
- Hop size $H=8$
- Overlap $=24$
- Window function: Hann

Log compression is applied:

$$
S_{\log}(t,f)=\log(1+S(t,f))
$$

All channel spectrograms are stacked to $(C,F,T)$ where $C=7$ for the full feature setting.

### CNN Architecture
The model has three convolutional blocks followed by a regression head:
- Block 1: Conv(7->32), BN, ReLU, Conv(32->32), BN, ReLU, MaxPool, Dropout2d
- Block 2: Conv(32->64), BN, ReLU, Conv(64->64), BN, ReLU, MaxPool, Dropout2d
- Block 3: Conv(64->128), BN, ReLU, AdaptiveAvgPool(4,4)
- Head: Flatten -> Linear(2048,256) -> ReLU -> Dropout -> Linear(256,64) -> ReLU -> Dropout -> Linear(64,1)

Total trainable parameters: 682,273

## How to Run
Step 1
```bash
pip install -r requirements.txt
```

Step 2
```bash
python src/01_data_collection.py
```

Step 3
```bash
python src/02_feature_engineering.py
```

Step 4
```bash
python src/03_spectrogram_generator.py
```

Step 5
```bash
python src/06_train.py
```

Step 6
```bash
python src/07_evaluate.py
```

Step 7
```bash
python src/08_ablation.py
```

## Reproducibility
- Config-driven experiment settings are managed in `config.yaml`
- Random seeds are fixed in training and ablation scripts
- Key outputs are saved to `data/processed/`, `figures/`, and `models/`

## Results

### Test Metrics (metrics.csv)
| Company | MSE | RMSE | MAE | MAPE | R2 | DirectionalAccuracy |
|---|---:|---:|---:|---:|---:|---:|
| RELIANCE.NS | 9073802020.781618 | 95256.506449 | 73030.074042 | 5.044303 | 0.460769 | 54.729730 |
| TCS.NS | 642584105744.464200 | 801613.439099 | 632152.206517 | 5.279589 | 0.176190 | 51.689189 |
| INFY.NS | 67748759467.157616 | 260285.918688 | 217532.847837 | 9.363170 | 0.165913 | 51.351351 |

### Ablation Results (ablation_results.csv)
| Experiment | Parameter | Value | RMSE | MAPE | DA |
|---|---|---|---:|---:|---:|
| WindowLength | window_length | 16 | 59.003105 | 3.420948 | 50.675676 |
| WindowLength | window_length | 32 | 110.616553 | 6.177357 | 56.081081 |
| WindowLength | window_length | 64 | 123.848049 | 7.086399 | 54.391892 |
| ForecastHorizon | forecast_horizon | 1 | 78.323368 | 4.711698 | 54.000000 |
| ForecastHorizon | forecast_horizon | 5 | 87.691336 | 5.274692 | 54.729730 |
| ForecastHorizon | forecast_horizon | 10 | 168.727545 | 9.897689 | 50.859107 |
| ForecastHorizon | forecast_horizon | 20 | 216.029003 | 12.948932 | 55.871886 |
| FeatureSet | feature_set | A | 75.858105 | 4.296529 | 51.013514 |
| FeatureSet | feature_set | B | 73.747189 | 4.268317 | 53.378378 |
| FeatureSet | feature_set | C | 75.824293 | 4.436028 | 54.729730 |

## Key Findings
- Best RMSE in window ablation occurred at $L=16$.
- Best directional accuracy in horizon ablation occurred at $\Delta t=20$.
- Feature Set B (close + technical indicators) improved over close-only baseline A.
- RELIANCE.NS was easiest to model overall (lowest RMSE, best $R^2$).
- TCS.NS showed the largest absolute errors, while INFY.NS had the highest relative error (MAPE).

## References
1. Y. Zhang and C. Aggarwal, Stock Market Prediction Using Deep Learning, IEEE Access.
2. A. Tsantekidis et al., Deep Learning for Financial Time Series Forecasting.
3. S. Hochreiter and J. Schmidhuber, Long Short-Term Memory, Neural Computation, 1997.
4. A. Borovykh et al., Conditional Time Series Forecasting with CNNs.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
