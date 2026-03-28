# pyright: reportGeneralTypeIssues=false, reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportMissingTypeArgument=false, reportMissingParameterType=false
import copy
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_stft_spectrogram(signal_1d: np.ndarray, signal_cfg: dict[str, Any]) -> np.ndarray:
    from scipy.signal import stft

    _, _, zxx = stft(
        signal_1d,
        nperseg=signal_cfg["window_length"],
        noverlap=signal_cfg["overlap"],
        window=signal_cfg["window_fn"],
        boundary=None,  # pyright: ignore[reportArgumentType]
        padded=False,
    )

    s = np.abs(zxx) ** 2
    if signal_cfg["log_scale"]:
        s = np.log1p(s)
    return s


def build_multichannel_spectrogram(df_window: pd.DataFrame, signal_cfg: dict[str, Any]) -> np.ndarray:
    channels = []
    for col in df_window.columns:
        spec = compute_stft_spectrogram(df_window[col].to_numpy(dtype=np.float64), signal_cfg)
        channels.append(spec)
    return np.stack(channels, axis=0)


def create_samples(
    df_features: pd.DataFrame,
    raw_close_series: pd.Series,
    signal_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = signal_cfg["sliding_window"]
    delta_t = signal_cfg["forecast_horizon"]

    raw_close_series = raw_close_series.reindex(df_features.index)

    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    d_list: list[Any] = []

    max_start = len(df_features) - w - delta_t
    if max_start < 0:
        return np.empty((0, df_features.shape[1], 0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype="datetime64[ns]")

    for i in range(max_start + 1):
        window_df = df_features.iloc[i : i + w]
        x_spec = build_multichannel_spectrogram(window_df, signal_cfg)

        target_idx = i + w + delta_t - 1
        y_val = raw_close_series.iloc[target_idx]
        if pd.isna(y_val):
            continue

        x_list.append(x_spec.astype(np.float32))
        y_list.append(float(y_val))
        d_list.append(df_features.index[target_idx])

    if len(x_list) == 0:
        return np.empty((0, df_features.shape[1], 0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype="datetime64[ns]")

    return np.stack(x_list, axis=0), np.asarray(y_list, dtype=np.float32), np.asarray(d_list, dtype="datetime64[ns]")


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float().view(-1, 1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class AblationCNN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)
        return x


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    denom = np.abs(y_true)
    mask = denom > 1e-12
    if np.any(mask):
        mape = float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)
    else:
        mape = float("nan")

    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    if len(true_direction) > 0:
        da = float(np.mean(true_direction == pred_direction) * 100.0)
    else:
        da = float("nan")

    return rmse, mape, da


def train_and_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    in_channels: int,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[float, float, float]:
    train_loader = DataLoader(ArrayDataset(x_train, y_train), batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(ArrayDataset(x_val, y_val), batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(ArrayDataset(x_test, y_test), batch_size=1, shuffle=False, num_workers=0)

    model = AblationCNN(in_channels=in_channels).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for _ in range(train_cfg["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(float(criterion(pred, yb).item()))

        mean_val = float(np.mean(val_losses)) if len(val_losses) > 0 else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step()

    model.load_state_dict(best_state)
    model.eval()

    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
            trues.extend(yb.detach().cpu().numpy().reshape(-1).tolist())

    y_pred = np.asarray(preds, dtype=np.float64)
    y_true = np.asarray(trues, dtype=np.float64)
    return compute_metrics(y_true, y_pred)


def run_single_setting(
    company: str,
    signal_cfg: dict[str, Any],
    feature_cols: list[str],
    in_channels: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[float, float, float]:
    processed_dir = ROOT_DIR / config["paths"]["processed"]
    raw_dir = ROOT_DIR / config["paths"]["raw_data"]

    train_df = pd.read_csv(processed_dir / f"{company}_train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv(processed_dir / f"{company}_val.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv(processed_dir / f"{company}_test.csv", index_col=0, parse_dates=True)

    train_df = train_df[feature_cols]
    val_df = val_df[feature_cols]
    test_df = test_df[feature_cols]

    raw_company = pd.read_csv(raw_dir / f"{company}.csv", index_col=0, parse_dates=True)
    raw_close = raw_company[company] if company in raw_company.columns else raw_company.iloc[:, 0]

    x_train, y_train, _ = create_samples(train_df, raw_close, signal_cfg)
    x_val, y_val, _ = create_samples(val_df, raw_close, signal_cfg)
    x_test, y_test, _ = create_samples(test_df, raw_close, signal_cfg)

    if min(len(x_train), len(x_val), len(x_test)) == 0:
        raise RuntimeError("One of the splits has zero samples. Adjust window/horizon settings.")

    train_cfg = dict(config["training"])
    train_cfg["epochs"] = 30

    return train_and_eval(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        in_channels,
        train_cfg,
        device,
    )


def save_ablation_figure(results_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    w_df = results_df[results_df["Experiment"] == "WindowLength"].sort_values("Value")
    axes[0].plot(w_df["Value"].to_numpy(dtype=float), w_df["RMSE"].to_numpy(dtype=float), marker="o", linewidth=2)
    axes[0].set_title("Window Length vs RMSE")
    axes[0].set_xlabel("Window Length")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.3)

    h_df = results_df[results_df["Experiment"] == "ForecastHorizon"].sort_values("Value")
    ax2 = axes[1]
    ax2b = ax2.twinx()
    ax2.plot(h_df["Value"].to_numpy(dtype=float), h_df["RMSE"].to_numpy(dtype=float), marker="o", color="tab:blue", label="RMSE")
    ax2b.plot(h_df["Value"].to_numpy(dtype=float), h_df["DA"].to_numpy(dtype=float), marker="s", color="tab:orange", label="DA")
    ax2.set_title("Forecast Horizon vs RMSE and DA")
    ax2.set_xlabel("Forecast Horizon")
    ax2.set_ylabel("RMSE", color="tab:blue")
    ax2b.set_ylabel("Directional Accuracy (%)", color="tab:orange")
    ax2.grid(alpha=0.3)

    f_df = results_df[results_df["Experiment"] == "FeatureSet"].copy()
    f_df["Value"] = pd.Categorical(f_df["Value"], categories=["A", "B", "C"], ordered=True)
    f_df = f_df.sort_values("Value")
    axes[2].bar(f_df["Value"].astype(str), f_df["RMSE"].to_numpy(dtype=float), color=["#4e79a7", "#f28e2b", "#59a14f"])
    axes[2].set_title("Feature Set vs RMSE")
    axes[2].set_xlabel("Feature Set")
    axes[2].set_ylabel("RMSE")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "ablation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    config = load_config(CONFIG_PATH)
    config["training"]["epochs"] = 30

    seed = int(config["training"].get("seed", 42))
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    company = config["data"]["companies"][0]
    print(f"Running ablation for first company only: {company}")

    results: list[dict[str, Any]] = []

    base_signal = dict(config["signal"])

    for l in [16, 32, 64]:
        signal_cfg = dict(base_signal)
        signal_cfg["window_length"] = l
        signal_cfg["overlap"] = l - 8

        rmse, mape, da = run_single_setting(
            company=company,
            signal_cfg=signal_cfg,
            feature_cols=["close", "log_return", "volatility", "RSI", "MACD", "sensex_close", "usd_inr"],
            in_channels=7,
            config=config,
            device=device,
        )

        results.append(
            {
                "Experiment": "WindowLength",
                "Parameter": "window_length",
                "Value": l,
                "RMSE": rmse,
                "MAPE": mape,
                "DA": da,
            }
        )

    for dt in [1, 5, 10, 20]:
        signal_cfg = dict(base_signal)
        signal_cfg["forecast_horizon"] = dt

        rmse, mape, da = run_single_setting(
            company=company,
            signal_cfg=signal_cfg,
            feature_cols=["close", "log_return", "volatility", "RSI", "MACD", "sensex_close", "usd_inr"],
            in_channels=7,
            config=config,
            device=device,
        )

        results.append(
            {
                "Experiment": "ForecastHorizon",
                "Parameter": "forecast_horizon",
                "Value": dt,
                "RMSE": rmse,
                "MAPE": mape,
                "DA": da,
            }
        )

    feature_set_configs = [
        ("A", ["close"], 1),
        ("B", ["close", "log_return", "volatility", "RSI"], 4),
        ("C", ["close", "log_return", "volatility", "RSI", "MACD", "sensex_close", "usd_inr"], 7),
    ]

    for tag, cols, in_channels in feature_set_configs:
        signal_cfg = dict(base_signal)

        rmse, mape, da = run_single_setting(
            company=company,
            signal_cfg=signal_cfg,
            feature_cols=cols,
            in_channels=in_channels,
            config=config,
            device=device,
        )

        results.append(
            {
                "Experiment": "FeatureSet",
                "Parameter": "feature_set",
                "Value": tag,
                "RMSE": rmse,
                "MAPE": mape,
                "DA": da,
            }
        )

    results_df = pd.DataFrame(results, columns=["Experiment", "Parameter", "Value", "RMSE", "MAPE", "DA"])

    processed_dir = ROOT_DIR / config["paths"]["processed"]
    figures_dir = ROOT_DIR / config["paths"]["figures"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(processed_dir / "ablation_results.csv", index=False)
    save_ablation_figure(results_df, figures_dir)

    print("\nAblation Results")
    print("=" * 88)
    print(results_df.to_string(index=False, justify="center", float_format=lambda x: f"{x:.6f}"))

    print("Phase 6 Complete - Ablation study done.")


if __name__ == "__main__":
    main()
