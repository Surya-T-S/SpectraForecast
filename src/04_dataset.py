# pyright: reportGeneralTypeIssues=false, reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportMissingTypeArgument=false, reportMissingParameterType=false
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[1]


class SpectrogramDataset(Dataset):
    def __init__(self, x_path: Union[str, Path], y_path: Union[str, Path]) -> None:
        x_np = np.load(x_path)
        y_np = np.load(y_path)

        self.x = torch.FloatTensor(x_np)
        self.y = torch.FloatTensor(y_np).view(-1, 1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def get_dataloaders(
    company: str,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    spectrogram_dir = ROOT_DIR / config["paths"]["spectrograms"]
    batch_size = config["training"]["batch_size"]

    x_train_path = spectrogram_dir / f"{company}_X_train.npy"
    y_train_path = spectrogram_dir / f"{company}_y_train.npy"
    x_val_path = spectrogram_dir / f"{company}_X_val.npy"
    y_val_path = spectrogram_dir / f"{company}_y_val.npy"
    x_test_path = spectrogram_dir / f"{company}_X_test.npy"
    y_test_path = spectrogram_dir / f"{company}_y_test.npy"

    train_dataset = SpectrogramDataset(x_train_path, y_train_path)
    val_dataset = SpectrogramDataset(x_val_path, y_val_path)
    test_dataset = SpectrogramDataset(x_test_path, y_test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
