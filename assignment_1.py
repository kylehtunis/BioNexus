import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

plt.style.use("ggplot")

DEFAULT_DATA_FILE = "sample_data/sample_data.csv"


@dataclass
class KineticResult:
    """Stores kinetic parameters from Lineweaver-Burk analysis."""

    condition: float | None
    km: float
    vmax: float
    r_squared: float
    model: LinearRegression
    reci_km: float
    x_reci: np.ndarray
    y_reci: np.ndarray


def load_data(filepath: str) -> pd.DataFrame:
    """Load kinetics data from CSV file."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"No such file as: {filepath}")
        sys.exit(1)


def compute_kinetics(s1: np.ndarray, rate: np.ndarray, condition: float | None = None) -> KineticResult:
    """Compute Km and Vmax using Lineweaver-Burk (double reciprocal) analysis."""
    x_reci = (1 / s1).reshape(-1, 1)
    y_reci = 1 / rate

    model = LinearRegression().fit(x_reci, y_reci)
    reci_km = -model.intercept_ / model.coef_[0]

    return KineticResult(
        condition=condition,
        km=1 / abs(reci_km),
        vmax=1 / abs(model.intercept_),
        r_squared=model.score(x_reci, y_reci),
        model=model,
        reci_km=reci_km,
        x_reci=x_reci,
        y_reci=y_reci,
    )


def print_results(results: list[KineticResult], multi_condition: bool = False) -> None:
    """Print kinetic parameters to console."""
    if multi_condition:
        print("\nAnalyzing multiple S2 conditions:\n")
        print(f"{'S2':<12} {'Km':<12} {'Vmax':<12} {'R²':<12}")
        print("-" * 48)
        for r in results:
            print(f"{r.condition:<12.4g} {r.km:<12.4f} {r.vmax:<12.4f} {r.r_squared:<12.4f}")
    else:
        r = results[0]
        print(f"\nKm: {r.km}")
        print(f"Vmax: {r.vmax}")
        print(f"Coefficient of determination: {r.r_squared}\n")


def plot_michaelis_menten(df: pd.DataFrame, results: list[KineticResult], output_file: str = "MM-plot.eps") -> None:
    """Create Michaelis-Menten plot (S vs v) with Km and Vmax/2 annotations."""
    fig, ax = plt.subplots()
    multi_condition = len(results) > 1

    if multi_condition:
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        for i, r in enumerate(results):
            subset = df[df["S2"] == r.condition]
            ax.scatter(subset["S1"], subset["Rate"], edgecolor="k", facecolor=colors[i], alpha=0.5, label=f"S2={r.condition}")
            ax.axhline(y=r.vmax / 2, color=colors[i], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(x=r.km, color=colors[i], linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_title("Michaelis-Menten Plot\n(dashed = Vmax/2, dotted = Km)")
        ax.legend(facecolor="white", title="Condition")
    else:
        r = results[0]
        ax.scatter(df["S1"], df["Rate"], edgecolor="k", facecolor="blue", alpha=0.5, label="Sample data")
        ax.axhline(y=r.vmax / 2, color="red", linestyle="--", linewidth=1, label=f"Vmax/2 = {r.vmax/2:.4f}")
        ax.axvline(x=r.km, color="green", linestyle=":", linewidth=1, label=f"Km = {r.km:.4f}")
        ax.set_title("Michaelis-Menten Plot")
        ax.legend(facecolor="white")

    ax.set_xlabel("[Substrate]")
    ax.set_ylabel("Reaction Rate")
    fig.savefig(output_file, format="eps")
    plt.show()


def plot_lineweaver_burk(df: pd.DataFrame, results: list[KineticResult], output_file: str = "Lineweaver-Burk.eps") -> None:
    """Create Lineweaver-Burk (double reciprocal) plot."""
    fig, ax = plt.subplots()
    multi_condition = len(results) > 1

    if multi_condition:
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        for i, r in enumerate(results):
            x_line = np.linspace(min(r.reci_km, r.x_reci.min()), r.x_reci.max(), 100).reshape(-1, 1)
            y_line = r.model.predict(x_line)
            ax.plot(x_line, y_line, color=colors[i], label=f"S2={r.condition}")
            ax.scatter(r.x_reci, r.y_reci, edgecolor="k", facecolor=colors[i], alpha=0.5)
        ax.legend(facecolor="white", title="Condition")
    else:
        r = results[0]
        y_pred = r.model.predict(r.x_reci)
        ax.plot([[r.reci_km], r.x_reci[0]], [0, y_pred[0]], c="k", linestyle="--")
        ax.plot(r.x_reci, y_pred, color="k", label="Regression model")
        ax.scatter(r.x_reci, r.y_reci, edgecolor="k", facecolor="blue", alpha=0.5, label="Sample data")
        ax.legend(facecolor="white")

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("1 / [Substrate]")
    ax.set_ylabel("1 / Enzyme Activity")
    ax.set_title("Lineweaver-Burk Plot")
    fig.savefig(output_file, format="eps")
    plt.show()


def analyze(df: pd.DataFrame) -> list[KineticResult]:
    """Analyze kinetics data, handling single or multiple S2 conditions."""
    if "S2" in df.columns:
        return [
            compute_kinetics(subset["S1"].values, subset["Rate"].values, condition=s2_val)
            for s2_val in df["S2"].unique()
            for subset in [df[df["S2"] == s2_val]]
        ]
    return [compute_kinetics(df["S1"].values, df["Rate"].values)]


def main() -> None:
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        print("No filename given, using 'sample_data.csv'")
        print("To use own data give the filename as an argument: python assignment_1.py 'filename'")
        filename = DEFAULT_DATA_FILE

    df = load_data(filename)

    if "S2" not in df.columns:
        print(df)

    results = analyze(df)
    print_results(results, multi_condition=len(results) > 1)
    plot_michaelis_menten(df, results)
    plot_lineweaver_burk(df, results)


if __name__ == "__main__":
    main()
