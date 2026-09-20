"""
Genera el grafico de resultados del entrenamiento (docs/img/training_curves.png)
a partir de los datos que train.py guarda en runs/ para TensorBoard.

Uso:
    python scripts/plot_training.py                       # usa la corrida mas reciente
    python scripts/plot_training.py --run runs/20260920-002049
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def series(acc: EventAccumulator, tag: str):
    events = acc.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    parser = argparse.ArgumentParser(description="Grafico de resultados del entrenamiento")
    parser.add_argument("--run", type=Path, help="carpeta de runs/ (por defecto la mas reciente)")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "img" / "training_curves.png")
    args = parser.parse_args()

    run = args.run or Path(max(glob.glob(str(ROOT / "runs" / "*")), key=os.path.getmtime))
    acc = EventAccumulator(str(run))
    acc.Reload()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    steps, win_rate = series(acc, "eval/win_rate_vs_random")
    baseline = acc.Scalars("eval/win_rate_random_baseline")[0].value
    axes[0].plot(steps, win_rate * 100, marker="o", markersize=3)
    axes[0].axhline(baseline * 100, color="gray", linestyle="--", label=f"random vs random ({baseline:.0%})")
    axes[0].set(title="Win rate vs a random opponent", xlabel="update", ylabel="win rate (%)", ylim=(0, 105))
    axes[0].legend(loc="lower right")

    steps, entropy = series(acc, "train/entropy")
    axes[1].plot(steps, entropy)
    axes[1].set(title="Policy entropy", xlabel="update", ylabel="entropy (both slots)")

    steps, returns = series(acc, "selfplay/mean_return_oficial")
    window = 10
    axes[2].plot(steps, returns, alpha=0.25, label="per update")
    axes[2].plot(steps[window - 1 :], moving_average(returns, window), label=f"{window}-update average")
    axes[2].axhline(0, color="gray", linestyle="--")
    axes[2].set(title="Self-play return of the official team", xlabel="update", ylabel="mean return per game")
    axes[2].legend(loc="lower right")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Guardado en {args.out} (corrida {run.name})")


if __name__ == "__main__":
    main()
