"""
Print a compact one-line training run summary.

Default input: training_log.json in project root.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def metric_delta(before: dict, after: dict, key: str) -> str:
    b = float(before.get(key, 0.0))
    a = float(after.get(key, 0.0))
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{a:.4f} ({sign}{d:.4f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="One-line experiment summary")
    parser.add_argument(
        "--log",
        type=str,
        default="training_log.json",
        help="Path to training_log.json",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = ROOT / log_path

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    data = load_json(log_path)
    before = data.get("val_metrics_before", {})
    after = data.get("val_metrics_after", {})

    model = data.get("base_model", "unknown_model")
    epochs = data.get("epochs", "?")
    batch = data.get("batch_size", "?")
    train_size = data.get("train_size", "?")
    elapsed_min = round(float(data.get("elapsed_sec", 0.0)) / 60.0, 1)

    summary = (
        f"model={model} | train={train_size} | ep={epochs} | bs={batch} | "
        f"acc@1={metric_delta(before, after, 'acc@1')} | "
        f"acc@3={metric_delta(before, after, 'acc@3')} | "
        f"acc@10={metric_delta(before, after, 'acc@10')} | "
        f"mrr@10={metric_delta(before, after, 'mrr@10')} | "
        f"time={elapsed_min}m"
    )
    print(summary)


if __name__ == "__main__":
    main()
