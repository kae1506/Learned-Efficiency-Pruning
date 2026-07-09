import os
import sys
import yaml
import torch
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.train import train_epoch, evaluate


def plot_metrics(history: dict, save_path: str = "experiments/latest/base_model/plot.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MNIST MLP Training", fontsize=13, fontweight="bold")

    ax1.plot(epochs, history["train_loss"], marker="o", label="Train", color="steelblue")
    ax1.plot(epochs, history["test_loss"], marker="o", label="Test", color="tomato")
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    train_pct = [a * 100 for a in history["train_acc"]]
    test_pct  = [a * 100 for a in history["test_acc"]]
    ax2.plot(epochs, train_pct, marker="o", label="Train", color="steelblue")
    ax2.plot(epochs, test_pct,  marker="o", label="Test",  color="tomato")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(bottom=min(min(train_pct), min(test_pct)) - 1, top=101)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def main(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_loaders(**cfg["data"])

    model = MLP(**cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"Epoch {epoch:>2} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f}, acc {test_acc:.4f}"
        )

    os.makedirs("experiments/checkpoints", exist_ok=True)
    ckpt_path = "experiments/checkpoints/mnist_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg["model"]}, ckpt_path)
    print(f"Saved model to {ckpt_path}")

    plot_metrics(history)


if __name__ == "__main__":
    main()
