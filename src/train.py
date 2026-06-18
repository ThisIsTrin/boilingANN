from dataclasses import dataclass
import numpy as np
import torch
import pickle
import json
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import BoilingANN
import os
import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    try:
        x_train = torch.tensor(np.load("data/x_train.npy"), dtype=torch.float32)
        y_train = torch.tensor(np.load("data/y_train.npy"), dtype=torch.float32)
        x_val = torch.tensor(np.load("data/x_val.npy"), dtype=torch.float32)
        y_val = torch.tensor(np.load("data/y_val.npy"), dtype=torch.float32)
        y_train_raw = np.load("data/y_train_raw.npy")
        y_val_raw = np.load("data/y_val_raw.npy")

        return x_train, y_train, x_val, y_val, y_train_raw, y_val_raw

    except FileNotFoundError as e:
        raise RuntimeError(f"File not found: {e}")


def load_utils():
    try:
        with open("models/scaler_y.pkl", "rb") as f:
            scaler_y = pickle.load(f)
        with open("models/feature_names.json") as f:
            feature_names = json.load(f)

        return scaler_y, feature_names

    except FileNotFoundError as e:
        raise RuntimeError(f"File not found {e}")


def dataloader(x_train, y_train):
    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    return train_loader


def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def predict_heat_flux(X_tensor, model, scaler_y):

    model.eval()

    with torch.no_grad():
        log_q_scaled = model(X_tensor).numpy()

    log_q = scaler_y.inverse_transform(log_q_scaled.reshape(-1, 1)).ravel()
    q = 10**log_q

    return q


def train_epoch(model, train_loader, criterion, optimizer, x_train):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in train_loader:
        # Forward
        optimizer.zero_grad()
        prediction = model(x_batch)
        loss = criterion(prediction, y_batch)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(x_batch)
    return total_loss / len(x_train)


def validate(model, criterion, x_val, y_val):
    model.eval()
    with torch.no_grad():
        prediction = model(x_val)
        val_loss = criterion(prediction, y_val)

    return val_loss.item()


def train():
    x_train, y_train, x_val, y_val, y_train_raw, y_val_raw = load_data()
    scaler_y, feature_names = load_utils()

    inputDIM = x_train.shape[1]

    train_loader = dataloader(x_train, y_train)

    model = BoilingANN(inputDIM)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=30, factor=0.5, min_lr=1e-6
    )

    EPOCHS = 2000
    EARLY_STOP_PATIENCE = 150
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    print("\nTraining...\n")

    try:
        for epoch in range(EPOCHS):

            train_loss = train_epoch(model, train_loader, criterion, optimizer, x_train)

            val_loss = validate(model, criterion, x_val, y_val)

            # Learning Rate
            scheduler.step(val_loss)

            # Tracking Losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "models/best_model.pt")
            else:
                patience_counter += 1

            # Early Stoping
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch+1}")
                break

            # Log
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1:5d} | "
                    f"Train {train_loss:.5f} | "
                    f"Val {val_loss:.5f} | "
                    f"Best {best_val_loss:.5f} | "
                    f"LR {optimizer.param_groups[0]['lr']:.2e}"
                )

            # print(f"\nBest val loss: {best_val_loss:.6f}")
    except KeyboardInterrupt:
        print("Interupt by user")
    except Exception as e:
        print(f"Training Fail: {e}")
        raise

    # Best model for evaluation
    model.load_state_dict(torch.load("models/best_model.pt"))

    val_pred_raw = predict_heat_flux(x_val, model, scaler_y)
    train_pred_raw = predict_heat_flux(x_train, model, scaler_y)

    val_mape = mape(y_val_raw, val_pred_raw)
    train_mape = mape(y_train_raw, train_pred_raw)

    print(f"Train MAPE : {train_mape:.2f}%")
    print(f"Val   MAPE : {val_mape:.2f}%  (target <10%)")
    # print(f"Automata baseline: 6.16-7.50%")

    # Check experimental test set if available, ai generated
    if os.path.exists("data/X_test.npy"):
        X_test = torch.tensor(np.load("data/X_test.npy"), dtype=torch.float32)
        y_test_raw = np.load("data/y_test_raw.npy")
        test_pred_raw = predict_heat_flux(X_test, model, scaler_y)
        test_mape = mape(y_test_raw, test_pred_raw)
        print(f"Test  MAPE : {test_mape:.2f}%  (target <15%)")
    else:
        test_pred_raw = None
        y_test_raw = None
        print("No experimental test set yet")

    _plot_results(train_losses, val_losses, y_val_raw, val_pred_raw, val_mape)


def _plot_results(train_losses, val_losses, y_val_raw, val_pred_raw, val_mape):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss curves
    axes[0].plot(train_losses, label="Train", alpha=0.8, linewidth=0.8)
    axes[0].plot(val_losses, label="Val", alpha=0.8, linewidth=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (log scale)")
    axes[0].set_yscale("log")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend()

    # Validation parity plot
    ax = axes[1]
    sc = ax.scatter(
        y_val_raw, val_pred_raw, c=np.log10(y_val_raw), cmap="viridis", alpha=0.4, s=10
    )
    plt.colorbar(sc, ax=ax, label="log10(q'' true)")
    lo = min(y_val_raw.min(), val_pred_raw.min()) * 0.9
    hi = max(y_val_raw.max(), val_pred_raw.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect")
    ax.plot([lo, hi], [x * 1.15 for x in [lo, hi]], "r--", lw=1, alpha=0.6)
    ax.plot(
        [lo, hi], [x * 0.85 for x in [lo, hi]], "r--", lw=1, alpha=0.6, label="±15%"
    )
    ax.set_xlabel("Simulation Q'' (Wcm2)")
    ax.set_ylabel("ANN predicted Q'' (Wcm2)")
    ax.set_title(f"Validation Parity\nMAPE = {val_mape:.2f}%")
    ax.legend(fontsize=8)

    # Residuals by feature (phi)
    ax = axes[2]
    residuals_pct = (val_pred_raw - y_val_raw) / y_val_raw * 100

    ax.scatter(val_pred_raw, residuals_pct, alpha=0.3, s=10, color="steelblue")
    ax.axhline(y=0, color="k", linestyle="--", lw=1.5)
    ax.axhline(y=15, color="r", linestyle="--", lw=1, alpha=0.6)
    ax.axhline(y=-15, color="r", linestyle="--", lw=1, alpha=0.6, label="±15%")
    ax.set_xlabel("ANN predicted Q'' (Wcm2)")
    ax.set_ylabel("Residual (%)")
    ax.set_title("Residuals vs Predicted")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/training_results.png", dpi=150)
    plt.show()
    print("Plot saved: results/training_results.png")

    pd.DataFrame(
        {
            "epoch": range(len(train_losses)),
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
    ).to_csv("results/loss_history.csv", index=False)


if __name__ == "__main__":
    train()
