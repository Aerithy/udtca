import argparse
import csv
import json
from pathlib import Path


def load_curve(run_dir: Path):
    curve_path = run_dir / "loss_curve.csv"
    steps = []
    losses = []
    with curve_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses


def load_summary(run_dir: Path):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare YOLO DDP runs")
    parser.add_argument("--run-a", type=str, required=True, help="Path to run directory A")
    parser.add_argument("--run-b", type=str, required=True, help="Path to run directory B")
    parser.add_argument("--out", type=str, default="comparison.png", help="Output plot path")
    args = parser.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)

    steps_a, losses_a = load_curve(run_a)
    steps_b, losses_b = load_curve(run_b)

    summary_a = load_summary(run_a)
    summary_b = load_summary(run_b)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib is required for plotting: {exc}")

    plt.figure(figsize=(8, 5))
    plt.plot(steps_a, losses_a, label=summary_a.get("run_name", "run_a"))
    plt.plot(steps_b, losses_b, label=summary_b.get("run_name", "run_b"))
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)

    print("Run A summary:")
    print(json.dumps(summary_a, indent=2))
    print("Run B summary:")
    print(json.dumps(summary_b, indent=2))
    print(f"Saved comparison plot to {args.out}")


if __name__ == "__main__":
    main()
