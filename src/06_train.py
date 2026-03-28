# pyright: reportGeneralTypeIssues=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false
import importlib.util
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml


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


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_company(company: str, config: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    train_loader, val_loader, _ = get_dataloaders(company, config)

    model = SpectrogramCNN().to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["patience"]
    best_epoch = 0

    train_losses: list[float] = []
    val_losses: list[float] = []

    models_dir = ROOT_DIR / config["paths"]["models"]
    figures_dir = ROOT_DIR / config["paths"]["figures"]
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"{company}_best.pth"

    total_epochs = config["training"]["epochs"]
    grad_clip = config["training"]["grad_clip"]

    for epoch in range(total_epochs):
        model.train()
        running_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_train_loss += loss.item()

        train_loss = running_train_loss / max(1, len(train_loader))

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                running_val_loss += loss.item()

        val_loss = running_val_loss / max(1, len(val_loader))

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{total_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), checkpoint_path)
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print(f"Training summary for {company}: best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.axvline(best_epoch, linestyle="--", color="red", label=f"Best Epoch: {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curve - {company}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"training_curve_{company}.png", dpi=200)
    plt.close()

    history: dict[str, Any] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

    return model, history


if __name__ == "__main__":
    config = load_config(CONFIG_PATH)

    for company in config["data"]["companies"]:
        print(f"\n{'=' * 20} {company} {'=' * 20}")
        train_company(company, config)

    print("Phase 4 Complete - All models trained.")
