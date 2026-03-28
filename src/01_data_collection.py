# pyright: reportGeneralTypeIssues=false, reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportMissingTypeArgument=false
from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_and_merge_data(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    raw_dir = ROOT_DIR / config["paths"]["raw_data"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    tickers = list(data_cfg["companies"]) + [data_cfg["market_index"], data_cfg["forex"]]
    downloaded = []

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=data_cfg["start_date"],
                end=data_cfg["end_date"],
                progress=False,
                auto_adjust=False,
            )

            if df is None:
                print(f"Warning: failed to download usable data for {ticker}; skipping.")
                continue

            if df.empty:
                print(f"Warning: failed to download usable data for {ticker}; skipping.")
                continue

            # yfinance can return either flat or multi-index columns.
            if "Close" in df.columns:
                close_series = df["Close"]
            elif isinstance(df.columns, pd.MultiIndex) and "Close" in df.columns.get_level_values(0):
                close_series = df["Close"]
            else:
                print(f"Warning: Close column missing for {ticker}; skipping.")
                continue

            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]

            close_series = pd.to_numeric(close_series, errors="coerce")
            close_df = close_series.to_frame(name=ticker)
            close_df.index.name = "Date"
            close_df.to_csv(raw_dir / f"{ticker}.csv")
            downloaded.append(close_df)
        except Exception as exc:
            print(f"Warning: failed to download {ticker} ({exc}); skipping.")

    if not downloaded:
        raise RuntimeError("No ticker data downloaded. Cannot proceed.")

    merged_df = pd.concat(downloaded, axis=1, join="inner")
    merged_df = merged_df.dropna(how="any")
    merged_df.index.name = "Date"

    merged_path = raw_dir / "merged_raw.csv"
    merged_df.to_csv(merged_path)

    assert len(merged_df) >= 1000, "Final merged dataframe has fewer than 1000 rows."

    print(f"Merged shape: {merged_df.shape}")
    print("First 5 rows:")
    print(merged_df.head())

    return merged_df


def main() -> None:
    config = load_config(CONFIG_PATH)
    fetch_and_merge_data(config)


if __name__ == "__main__":
    main()
