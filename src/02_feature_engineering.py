# pyright: reportGeneralTypeIssues=false, reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportMissingTypeArgument=false
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_indices(n_rows: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= n_rows:
        raise ValueError("Invalid split boundaries generated from config ratios.")
    return train_end, val_end


def build_feature_dataframe(
    merged_df: pd.DataFrame,
    company: str,
    market_index: str,
    forex: str,
) -> pd.DataFrame:
    close = merged_df[company]
    log_return = pd.Series(np.log(close / close.shift(1)), index=close.index, name="log_return")
    volatility = log_return.rolling(window=10).std()
    rsi = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close, window_fast=12, window_slow=26, window_sign=9).macd()
    rolling_mean_30 = close.rolling(window=30).mean()

    # Keep the final model input matrix to the exact 7 requested columns.
    _ = rolling_mean_30

    feature_df = pd.DataFrame(
        {
            "close": close,
            "log_return": log_return,
            "volatility": volatility,
            "RSI": rsi,
            "MACD": macd,
            "sensex_close": merged_df[market_index],
            "usd_inr": merged_df[forex],
        },
        index=merged_df.index,
    )

    return feature_df.dropna(how="any")


def normalize_and_split(
    feature_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler, int, int]:
    train_end, val_end = split_indices(len(feature_df), train_ratio, val_ratio)

    train_df = feature_df.iloc[:train_end].copy()
    val_df = feature_df.iloc[train_end:val_end].copy()
    test_df = feature_df.iloc[val_end:].copy()

    scaler = MinMaxScaler()
    scaler.fit(train_df.values)

    train_scaled = pd.DataFrame(
        scaler.transform(train_df.values),
        columns=feature_df.columns,
        index=train_df.index,
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_df.values),
        columns=feature_df.columns,
        index=val_df.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df.values),
        columns=feature_df.columns,
        index=test_df.index,
    )

    return train_scaled, val_scaled, test_scaled, scaler, train_end, val_end


def plot_time_series(
    merged_df: pd.DataFrame,
    companies: list[str],
    boundary_dates: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(companies), 1, figsize=(14, 10), sharex=True)
    if len(companies) == 1:
        axes = [axes]

    for ax, company in zip(axes, companies):
        ax.plot(merged_df.index, merged_df[company], label=f"{company} close", linewidth=1.0)
        train_boundary, val_boundary = boundary_dates[company]

        if train_boundary is not None:
            ax.axvline(train_boundary, linestyle="--", color="tab:green", linewidth=1.0)
        if val_boundary is not None:
            ax.axvline(val_boundary, linestyle="--", color="tab:red", linewidth=1.0)

        ax.set_title(f"{company} Close Price")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Raw Close Prices with Time-Based Split Boundaries", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_fft_spectrum(
    feature_frames: dict[str, pd.DataFrame],
    companies: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(companies), 1, figsize=(14, 10), sharex=False)
    if len(companies) == 1:
        axes = [axes]

    for ax, company in zip(axes, companies):
        close_values = feature_frames[company]["close"].to_numpy()
        n = len(close_values)

        freqs = np.fft.rfftfreq(n, d=1.0)
        magnitude = np.abs(np.fft.rfft(close_values))
        magnitude = np.maximum(magnitude, 1e-10)

        ax.plot(freqs[1:], magnitude[1:], linewidth=1.0)
        ax.set_yscale("log")
        ax.set_title(f"FFT Spectrum - {company}")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude (log scale)")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_feature_engineering() -> None:
    config = load_config(CONFIG_PATH)

    data_cfg = config["data"]
    paths_cfg = config["paths"]

    raw_path = ROOT_DIR / paths_cfg["raw_data"] / "merged_raw.csv"
    processed_dir = ROOT_DIR / paths_cfg["processed"]
    figures_dir = ROOT_DIR / paths_cfg["figures"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    merged_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    companies = list(data_cfg["companies"])
    market_index = data_cfg["market_index"]
    forex = data_cfg["forex"]

    required_cols = companies + [market_index, forex]
    missing_cols = [col for col in required_cols if col not in merged_df.columns]
    if missing_cols:
        raise KeyError(f"Merged raw data is missing expected columns: {missing_cols}")

    feature_frames: dict[str, pd.DataFrame] = {}
    boundary_dates: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]] = {}
    report_rows: dict[str, dict[str, int]] = {}
    train_minmax_report: dict[str, pd.DataFrame] = {}

    for company in companies:
        feature_df = build_feature_dataframe(merged_df, company, market_index, forex)
        feature_frames[company] = feature_df

        feature_df.to_csv(processed_dir / f"{company}_features.csv", index=True)

        train_scaled, val_scaled, test_scaled, scaler, train_end, val_end = normalize_and_split(
            feature_df,
            data_cfg["train_ratio"],
            data_cfg["val_ratio"],
        )

        joblib.dump(scaler, processed_dir / f"{company}_scaler.pkl")

        train_scaled.to_csv(processed_dir / f"{company}_train.csv", index=True)
        val_scaled.to_csv(processed_dir / f"{company}_val.csv", index=True)
        test_scaled.to_csv(processed_dir / f"{company}_test.csv", index=True)

        train_boundary = feature_df.index[train_end - 1] if train_end > 0 else None
        val_boundary = feature_df.index[val_end - 1] if val_end > 0 else None
        boundary_dates[company] = (train_boundary, val_boundary)

        report_rows[company] = {
            "train_rows": len(train_scaled),
            "val_rows": len(val_scaled),
            "test_rows": len(test_scaled),
        }

        train_minmax_report[company] = pd.DataFrame(
            {
                "min": train_scaled.min(axis=0),
                "max": train_scaled.max(axis=0),
            }
        )

    plot_time_series(
        merged_df=merged_df,
        companies=companies,
        boundary_dates=boundary_dates,
        output_path=figures_dir / "time_series.png",
    )

    plot_fft_spectrum(
        feature_frames=feature_frames,
        companies=companies,
        output_path=figures_dir / "fft_spectrum.png",
    )

    print("\nFinal Report")
    print("=" * 60)

    for company in companies:
        print(f"\nCompany: {company}")
        print(
            f"Split rows -> train: {report_rows[company]['train_rows']}, "
            f"val: {report_rows[company]['val_rows']}, "
            f"test: {report_rows[company]['test_rows']}"
        )
        print(f"Feature columns: {list(feature_frames[company].columns)}")
        print("Train split min/max after normalization:")
        print(train_minmax_report[company])


def run_data_collection_first() -> None:
    script_path = Path(__file__).resolve().parent / "01_data_collection.py"
    subprocess.run([sys.executable, str(script_path)], check=True)


if __name__ == "__main__":
    run_data_collection_first()
    run_feature_engineering()
    print("Phase 1 Complete - Data ready.")
