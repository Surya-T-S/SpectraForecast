# pyright: reportGeneralTypeIssues=false, reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportMissingTypeArgument=false
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import stft


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_stft_spectrogram(signal_1d: np.ndarray, config: dict) -> np.ndarray:
    _, _, zxx = stft(
        signal_1d,
        nperseg=config["signal"]["window_length"],
        noverlap=config["signal"]["overlap"],
        window=config["signal"]["window_fn"],
        boundary=None,  # pyright: ignore[reportArgumentType]
        padded=False,
    )

    s = np.abs(zxx) ** 2

    if config["signal"]["log_scale"]:
        s = np.log1p(s)

    return s


def build_multichannel_spectrogram(df_features: pd.DataFrame, config: dict) -> np.ndarray:
    channels = []

    for col in df_features.columns:
        signal_1d = df_features[col].to_numpy(dtype=np.float64)
        spec = compute_stft_spectrogram(signal_1d, config)
        channels.append(spec)

    return np.stack(channels, axis=0)


def create_samples(
    df_features: pd.DataFrame,
    series_raw_close: pd.Series,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    w = config["signal"]["sliding_window"]
    delta_t = config["signal"]["forecast_horizon"]

    # Align raw close with split index to avoid date mismatch in targets.
    series_raw_close = series_raw_close.reindex(df_features.index)

    x_list = []
    y_list = []

    max_start = len(df_features) - w - delta_t
    for i in range(max_start + 1):
        window_df = df_features.iloc[i : i + w]
        x_spec = build_multichannel_spectrogram(window_df, config)

        y_val = series_raw_close.iloc[i + w + delta_t - 1]
        if pd.isna(y_val):
            continue

        x_list.append(x_spec)
        y_list.append(float(y_val))

    if len(x_list) == 0:
        return np.empty((0, df_features.shape[1], 0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    x_arr = np.stack(x_list, axis=0).astype(np.float32)
    y_arr = np.asarray(y_list, dtype=np.float32)

    return x_arr, y_arr


def load_split_features(processed_dir: Path, company: str, split: str) -> pd.DataFrame:
    split_path = processed_dir / f"{company}_{split}.csv"
    return pd.read_csv(split_path, index_col=0, parse_dates=True)


def load_raw_close_series(raw_dir: Path, company: str) -> pd.Series:
    raw_path = raw_dir / f"{company}.csv"
    raw_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    if company in raw_df.columns:
        series = raw_df[company]
    else:
        # Fallback for unexpected column naming while still using first close series.
        series = raw_df.iloc[:, 0]

    return series


def save_company_arrays(
    company: str,
    spectrogram_dir: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    np.save(spectrogram_dir / f"{company}_X_train.npy", x_train)
    np.save(spectrogram_dir / f"{company}_y_train.npy", y_train)
    np.save(spectrogram_dir / f"{company}_X_val.npy", x_val)
    np.save(spectrogram_dir / f"{company}_y_val.npy", y_val)
    np.save(spectrogram_dir / f"{company}_X_test.npy", x_test)
    np.save(spectrogram_dir / f"{company}_y_test.npy", y_test)

    print(f"{company}_X_train.npy shape: {x_train.shape}")
    print(f"{company}_y_train.npy shape: {y_train.shape}")
    print(f"{company}_X_val.npy shape: {x_val.shape}")
    print(f"{company}_y_val.npy shape: {y_val.shape}")
    print(f"{company}_X_test.npy shape: {x_test.shape}")
    print(f"{company}_y_test.npy shape: {y_test.shape}")


def plot_first_training_spectrograms(
    companies: list[str],
    company_train_samples: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(companies), 1, figsize=(10, 12))
    if len(companies) == 1:
        axes = [axes]

    for ax, company in zip(axes, companies):
        x_train = company_train_samples[company]

        if x_train.shape[0] == 0:
            ax.text(0.5, 0.5, "No training samples", ha="center", va="center")
            ax.set_title(company)
            ax.set_axis_off()
            continue

        img = ax.imshow(x_train[0, 0], aspect="auto", origin="lower")
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Log Energy")
        ax.set_title(company)
        ax.set_xlabel("Time Bins")
        ax.set_ylabel("Frequency Bins")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    config = load_config(CONFIG_PATH)

    companies = list(config["data"]["companies"])

    raw_dir = ROOT_DIR / config["paths"]["raw_data"]
    processed_dir = ROOT_DIR / config["paths"]["processed"]
    spectrogram_dir = ROOT_DIR / config["paths"]["spectrograms"]
    figures_dir = ROOT_DIR / config["paths"]["figures"]

    spectrogram_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    company_train_samples: dict[str, np.ndarray] = {}

    for company in companies:
        train_df = load_split_features(processed_dir, company, "train")
        val_df = load_split_features(processed_dir, company, "val")
        test_df = load_split_features(processed_dir, company, "test")

        raw_close = load_raw_close_series(raw_dir, company)

        x_train, y_train = create_samples(train_df, raw_close, config)
        x_val, y_val = create_samples(val_df, raw_close, config)
        x_test, y_test = create_samples(test_df, raw_close, config)

        save_company_arrays(
            company=company,
            spectrogram_dir=spectrogram_dir,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )

        company_train_samples[company] = x_train

    plot_first_training_spectrograms(
        companies=companies,
        company_train_samples=company_train_samples,
        output_path=figures_dir / "spectrograms.png",
    )

    print("Phase 2 Complete - Spectrograms saved.")


if __name__ == "__main__":
    main()
