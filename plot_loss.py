"""
plot_loss.py

讀取 outputs/tuwei_lora/trainer_state.json，畫出 Training / Validation Loss，
輸出 images/loss_curve.png。
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATE_FILE = Path("outputs/tuwei_lora/trainer_state.json")
PLOT_FILE = Path("images/loss_curve.png")


def points(logs, key):
    return [(item["step"], item[key]) for item in logs if "step" in item and key in item]


def main():
    if not STATE_FILE.is_file():
        raise FileNotFoundError(f"找不到 {STATE_FILE}，請先執行 python train_qlora.py")

    logs = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("log_history", [])
    train = points(logs, "loss")
    valid = points(logs, "eval_loss")
    if not train and not valid:
        raise RuntimeError("log_history 裡沒有 loss 或 eval_loss")

    plt.figure(figsize=(10, 6))
    if train:
        x, y = zip(*train)
        plt.plot(x, y, label="Training Loss", linewidth=1.8)
    if valid:
        x, y = zip(*valid)
        plt.plot(x, y, "o-", label="Validation Loss", linewidth=1.8)
    plt.xlabel("Optimizer Step")
    plt.ylabel("Loss")
    plt.title("QLoRA Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Training loss points：{len(train)}，Validation loss points：{len(valid)}")
    print(f"已輸出：{PLOT_FILE.resolve()}")


if __name__ == "__main__":
    main()
