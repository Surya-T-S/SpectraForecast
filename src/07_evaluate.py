# pyright: reportGeneralTypeIssues=false, reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportMissingTypeArgument=false, reportMissingParameterType=false
import importlib.util
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_module = _load_module(Path(__file__).resolve().parent / "04_dataset.py", "dataset_module")
model_module = _load_module(Path(__file__).resolve().parent / "05_model.py", "model_module")

get_dataloaders = dataset_module.get_dataloaders
SpectrogramCNN = model_module.SpectrogramCNN


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)

    denom = np.abs(y_true)
    nonzero_mask = denom > 1e-12
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]) / denom[nonzero_mask]) * 100.0)
    else:
        mape = float("nan")

    r2 = r2_score(y_true, y_pred)

    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    if len(true_direction) > 0:
        directional_accuracy = float(np.mean(true_direction == pred_direction) * 100.0)
    else:
        directional_accuracy = float("nan")

    return {
        "MSE": float(mse),
        "RMSE": rmse,
        "MAE": float(mae),
        "MAPE": mape,
        "R2": float(r2),
        "DirectionalAccuracy": directional_accuracy,
    }


def inverse_transform_close(scaler: Any, values_norm: np.ndarray, n_features: int) -> np.ndarray:
    temp = np.zeros((len(values_norm), n_features), dtype=np.float64)
    temp[:, 0] = values_norm
    restored = scaler.inverse_transform(temp)
    return restored[:, 0]


def print_metrics_table(company: str, metrics: dict[str, float]) -> None:
    print(f"\nMetrics for {company}")
    print("-" * 54)
    print(f"{'Metric':<22}{'Value':>32}")
    print("-" * 54)
    for key in ["MSE", "RMSE", "MAE", "MAPE", "R2", "DirectionalAccuracy"]:
        print(f"{key:<22}{metrics[key]:>32.6f}")
    print("-" * 54)


def save_company_plots(
    company: str,
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figures_dir: Path,
) -> None:
    residuals = y_pred - y_true
    abs_err = np.abs(residuals)

    fig, axes = plt.subplots(3, 1, figsize=(13, 14))

    axes[0].plot(dates, y_true, color="tab:blue", linewidth=1.8, label="Actual")
    axes[0].plot(dates, y_pred, color="tab:orange", linestyle="--", linewidth=1.8, label="Predicted")
    axes[0].fill_between(dates, y_true, y_pred, color="gray", alpha=0.2, label="Error")
    axes[0].set_title(f"Actual vs Predicted Close Price - {company}")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Close Price")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    scatter = axes[1].scatter(y_true, y_pred, c=abs_err, cmap="viridis", alpha=0.85, edgecolors="none")
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    axes[1].plot([min_v, max_v], [min_v, max_v], color="red", linestyle="--", linewidth=1.5, label="Perfect Prediction")
    axes[1].set_title(f"Actual vs Predicted Scatter - {company}")
    axes[1].set_xlabel("Actual Price")
    axes[1].set_ylabel("Predicted Price")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    cbar = fig.colorbar(scatter, ax=axes[1])
    cbar.set_label("Absolute Error")

    bar_colors = ["green" if r > 0 else "red" for r in residuals]
    axes[2].bar(dates, residuals, color=bar_colors, alpha=0.75)
    axes[2].axhline(0.0, color="black", linewidth=1.2)
    axes[2].set_title(f"Residuals Over Time - {company}")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Residual (Pred - Actual)")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / f"predictions_{company}.png", dpi=200)
    plt.close(fig)


def save_metrics_comparison(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    companies = metrics_df["Company"].tolist()
    rmse_vals = metrics_df["RMSE"].to_numpy(dtype=float)
    mape_vals = metrics_df["MAPE"].to_numpy(dtype=float)

    x = np.arange(len(companies))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, rmse_vals, width, label="RMSE", color="tab:blue")
    bars2 = ax.bar(x + width / 2, mape_vals, width, label="MAPE", color="tab:orange")

    ax.set_title("RMSE and MAPE Comparison Across Companies")
    ax.set_xlabel("Company")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(x)
    ax.set_xticklabels(companies)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(figures_dir / "metrics_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    config = load_config(CONFIG_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    companies = list(config["data"]["companies"])
    processed_dir = ROOT_DIR / config["paths"]["processed"]
    figures_dir = ROOT_DIR / config["paths"]["figures"]
    models_dir = ROOT_DIR / config["paths"]["models"]

    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    for company in companies:
        model = SpectrogramCNN().to(device)
        checkpoint_path = models_dir / f"{company}_best.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        _, _, test_loader = get_dataloaders(company, config)

        model.eval()
        preds_norm: list[float] = []
        y_true_norm: list[float] = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_hat = model(x_batch)

                preds_norm.extend(y_hat.detach().cpu().numpy().reshape(-1).tolist())
                y_true_norm.extend(y_batch.detach().cpu().numpy().reshape(-1).tolist())

        preds_norm_arr = np.asarray(preds_norm, dtype=np.float64)
        y_true_norm_arr = np.asarray(y_true_norm, dtype=np.float64)

        scaler = joblib.load(processed_dir / f"{company}_scaler.pkl")
        n_features = int(getattr(scaler, "n_features_in_", 7))

        y_pred_inv = inverse_transform_close(scaler, preds_norm_arr, n_features)
        y_true_inv = inverse_transform_close(scaler, y_true_norm_arr, n_features)

        metrics = compute_metrics(y_true_inv, y_pred_inv)
        print_metrics_table(company, metrics)

        test_df = pd.read_csv(processed_dir / f"{company}_test.csv", index_col=0, parse_dates=True)
        w = config["signal"]["sliding_window"]
        delta_t = config["signal"]["forecast_horizon"]
        start_idx = w + delta_t - 1
        target_dates = test_df.index[start_idx : start_idx + len(y_true_inv)]

        save_company_plots(
            company=company,
            dates=target_dates.to_numpy(),
            y_true=y_true_inv,
            y_pred=y_pred_inv,
            figures_dir=figures_dir,
        )

        all_rows.append(
            {
                "Company": company,
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"],
                "R2": metrics["R2"],
                "DirectionalAccuracy": metrics["DirectionalAccuracy"],
            }
        )

    metrics_df = pd.DataFrame(
        all_rows,
        columns=["Company", "MSE", "RMSE", "MAE", "MAPE", "R2", "DirectionalAccuracy"],
    )
    metrics_df.to_csv(processed_dir / "metrics.csv", index=False)

    save_metrics_comparison(metrics_df, figures_dir)

    print("Phase 5 Complete - Evaluation done.")


if __name__ == "__main__":
    main()
