import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

plt.style.use("ggplot")

DEFAULT_DATA_FILE = "Kinetics.csv"

######## Part 1 ########

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


def compute_kinetics(
    s1: np.ndarray, rate: np.ndarray, condition: float | None = None
) -> KineticResult:
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


def ping_pong_model(substrates, vmax, km1, km2):
    """Ping-Pong Bi-Bi: v = Vmax*[S1]*[S2] / (Km2*[S1] + Km1*[S2] + [S1]*[S2])"""
    s1, s2 = substrates
    return vmax * s1 * s2 / (km2 * s1 + km1 * s2 + s1 * s2)


def sequential_model(substrates, vmax, km1, km2, ks1):
    """Sequential (Ternary Complex): v = Vmax*[S1]*[S2] / (Ks1*Km2 + Km2*[S1] + Km1*[S2] + [S1]*[S2])"""
    s1, s2 = substrates
    return vmax * s1 * s2 / (ks1 * km2 + km2 * s1 + km1 * s2 + s1 * s2)


def identify_mechanism(df: pd.DataFrame) -> dict:
    """Fit ping-pong and sequential models, compare with R² and Chi-squared."""
    s1 = df["S1"].values
    s2 = df["S2"].values
    rate = df["Rate"].values

    ss_tot = np.sum((rate - np.mean(rate)) ** 2)

    # Fit Ping-Pong model
    popt_pp, _ = curve_fit(
        ping_pong_model, (s1, s2), rate, p0=[1.0, 0.1, 0.1], maxfev=10000
    )
    pred_pp = ping_pong_model((s1, s2), *popt_pp)
    ss_res_pp = np.sum((rate - pred_pp) ** 2)
    r2_pp = 1 - ss_res_pp / ss_tot
    chi2_pp = np.sum((rate - pred_pp) ** 2 / np.abs(pred_pp))

    # Fit Sequential model
    popt_seq, _ = curve_fit(
        sequential_model, (s1, s2), rate, p0=[1.0, 0.1, 0.1, 0.1], maxfev=10000
    )
    pred_seq = sequential_model((s1, s2), *popt_seq)
    ss_res_seq = np.sum((rate - pred_seq) ** 2)
    r2_seq = 1 - ss_res_seq / ss_tot
    chi2_seq = np.sum((rate - pred_seq) ** 2 / np.abs(pred_seq))

    # Check LB slope parallelism
    slopes = []
    for s2_val in sorted(df["S2"].unique()):
        subset = df[df["S2"] == s2_val]
        x = (1 / subset["S1"].values).reshape(-1, 1)
        y = 1 / subset["Rate"].values
        m = LinearRegression().fit(x, y)
        slopes.append((s2_val, m.coef_[0]))

    slope_values = [s[1] for s in slopes]
    slope_cv = np.std(slope_values) / np.mean(slope_values)  # coefficient of variation

    result = {
        "pp_params": {"Vmax": popt_pp[0], "Km1": popt_pp[1], "Km2": popt_pp[2]},
        "pp_r2": r2_pp,
        "pp_chi2": chi2_pp,
        "seq_params": {
            "Vmax": popt_seq[0],
            "Km1": popt_seq[1],
            "Km2": popt_seq[2],
            "Ks1": popt_seq[3],
        },
        "seq_r2": r2_seq,
        "seq_chi2": chi2_seq,
        "slopes": slopes,
        "slope_cv": slope_cv,
    }
    return result


def print_mechanism(mech: dict) -> None:
    """Print mechanism identification results."""
    print("\n" + "=" * 60)
    print("MECHANISM IDENTIFICATION (Sub-question 1)")
    print("=" * 60)

    print("\n--- Ping-Pong Bi-Bi Model ---")
    print(f"  Vmax = {mech['pp_params']['Vmax']:.4f}")
    print(f"  Km1 (S1) = {mech['pp_params']['Km1']:.4f}")
    print(f"  Km2 (S2) = {mech['pp_params']['Km2']:.4f}")
    print(f"  R² = {mech['pp_r2']:.6f}")
    print(f"  Chi² = {mech['pp_chi2']:.6f}")

    print("\n--- Sequential (Ternary Complex) Model ---")
    print(f"  Vmax = {mech['seq_params']['Vmax']:.4f}")
    print(f"  Km1 (S1) = {mech['seq_params']['Km1']:.4f}")
    print(f"  Km2 (S2) = {mech['seq_params']['Km2']:.4f}")
    print(f"  Ks1 = {mech['seq_params']['Ks1']:.6f}")
    print(f"  R² = {mech['seq_r2']:.6f}")
    print(f"  Chi² = {mech['seq_chi2']:.6f}")

    print("\n--- Lineweaver-Burk Slope Analysis ---")
    print(f"  {'S2':<12} {'LB Slope':<15}")
    print("  " + "-" * 27)
    for s2_val, slope in mech["slopes"]:
        print(f"  {s2_val:<12.4g} {slope:<15.6f}")
    print(f"  Slope CV = {mech['slope_cv']:.4f}")
    if mech["slope_cv"] < 0.05:
        print("  -> Slopes are approximately parallel (consistent with Ping-Pong)")
    else:
        print("  -> Slopes vary with [S2] (consistent with Sequential mechanism)")

    # Determine best model
    if mech["pp_r2"] > mech["seq_r2"] - 0.001 and mech["pp_chi2"] < mech["seq_chi2"]:
        best = "Ping-Pong Bi-Bi"
    elif mech["seq_r2"] > mech["pp_r2"] + 0.001:
        best = "Sequential (Ternary Complex)"
    else:
        best = (
            "Ping-Pong Bi-Bi"
            if mech["pp_chi2"] <= mech["seq_chi2"]
            else "Sequential (Ternary Complex)"
        )

    print(f"\n  CONCLUSION: Europase follows a {best} mechanism.")
    print("=" * 60)


def compute_batch_reactor_time(km: float, vmax: float) -> dict:
    """Calculate batch reactor time to deplete S1 from 100 g/L to 1 g/L.

    With S2 in excess, rate = Vmax*[S1]/(Km+[S1]).
    Integrated: t = (Km * ln(S0/Sf) + (S0 - Sf)) / Vmax
    """
    mw_s1 = 150.0  # g/mol
    s0_gl = 100.0  # g/L
    sf_gl = 1.0  # g/L

    s0 = s0_gl / mw_s1 * 1000  # mM
    sf = sf_gl / mw_s1 * 1000  # mM

    t_seconds = (km * np.log(s0 / sf) + (s0 - sf)) / vmax

    return {
        "s0_mM": s0,
        "sf_mM": sf,
        "km": km,
        "vmax": vmax,
        "t_seconds": t_seconds,
        "t_minutes": t_seconds / 60,
    }


def print_batch_reactor_time(result: dict) -> None:
    """Print batch reactor time calculation."""
    print("\n" + "=" * 60)
    print("BATCH REACTOR TIME (Sub-question 3)")
    print("=" * 60)
    print(f"  S1 initial: 100 g/L = {result['s0_mM']:.2f} mM")
    print(f"  S1 final:   1 g/L   = {result['sf_mM']:.2f} mM")
    print(f"  Km (S2 excess) = {result['km']:.4f} mM")
    print(f"  Vmax (S2 excess) = {result['vmax']:.4f} mM/s")
    print(f"\n  Time = (Km * ln(S0/Sf) + (S0 - Sf)) / Vmax")
    print(f"  Time = {result['t_seconds']:.2f} seconds")
    print(f"       = {result['t_minutes']:.2f} minutes")
    print("=" * 60)


def print_results(results: list[KineticResult], multi_condition: bool = False) -> None:
    """Print kinetic parameters to console."""
    if multi_condition:
        print("\nAnalyzing multiple S2 conditions:\n")
        print(f"{'S2':<12} {'Km':<12} {'Vmax':<12} {'R²':<12}")
        print("-" * 48)
        for r in results:
            print(
                f"{r.condition:<12.4g} {r.km:<12.4f} {r.vmax:<12.4f} {r.r_squared:<12.4f}"
            )
    else:
        r = results[0]
        print(f"\nKm: {r.km}")
        print(f"Vmax: {r.vmax}")
        print(f"Coefficient of determination: {r.r_squared}\n")


def plot_michaelis_menten(
    df: pd.DataFrame, results: list[KineticResult], output_file: str = "MM-plot.eps"
) -> None:
    """Create Michaelis-Menten plot (S vs v) with Km and Vmax/2 annotations."""
    fig, ax = plt.subplots()
    multi_condition = len(results) > 1

    if multi_condition:
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        for i, r in enumerate(results):
            subset = df[df["S2"] == r.condition]
            ax.scatter(
                subset["S1"],
                subset["Rate"],
                edgecolor="k",
                facecolor=colors[i],
                alpha=0.5,
                label=f"S2={r.condition}",
            )
            ax.axhline(
                y=r.vmax / 2, color=colors[i], linestyle="--", linewidth=0.8, alpha=0.7
            )
            ax.axvline(x=r.km, color=colors[i], linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_title("Michaelis-Menten Plot\n(dashed = Vmax/2, dotted = Km)")
        ax.legend(facecolor="white", title="Condition")
    else:
        r = results[0]
        ax.scatter(
            df["S1"],
            df["Rate"],
            edgecolor="k",
            facecolor="blue",
            alpha=0.5,
            label="Sample data",
        )
        ax.axhline(
            y=r.vmax / 2,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Vmax/2 = {r.vmax/2:.4f}",
        )
        ax.axvline(
            x=r.km, color="green", linestyle=":", linewidth=1, label=f"Km = {r.km:.4f}"
        )
        ax.set_title("Michaelis-Menten Plot")
        ax.legend(facecolor="white")

    ax.set_xlabel("[Substrate]")
    ax.set_ylabel("Reaction Rate")
    fig.savefig(output_file, format="eps")
    plt.show()


def plot_lineweaver_burk(
    df: pd.DataFrame,
    results: list[KineticResult],
    output_file: str = "Lineweaver-Burk.eps",
) -> None:
    """Create Lineweaver-Burk (double reciprocal) plot."""
    fig, ax = plt.subplots()
    multi_condition = len(results) > 1

    if multi_condition:
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        for i, r in enumerate(results):
            x_line = np.linspace(
                min(r.reci_km, r.x_reci.min()), r.x_reci.max(), 100
            ).reshape(-1, 1)
            y_line = r.model.predict(x_line)
            ax.plot(x_line, y_line, color=colors[i], label=f"S2={r.condition}")
            ax.scatter(
                r.x_reci, r.y_reci, edgecolor="k", facecolor=colors[i], alpha=0.5
            )
        ax.legend(facecolor="white", title="Condition")
    else:
        r = results[0]
        y_pred = r.model.predict(r.x_reci)
        ax.plot([[r.reci_km], r.x_reci[0]], [0, y_pred[0]], c="k", linestyle="--")
        ax.plot(r.x_reci, y_pred, color="k", label="Regression model")
        ax.scatter(
            r.x_reci,
            r.y_reci,
            edgecolor="k",
            facecolor="blue",
            alpha=0.5,
            label="Sample data",
        )
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
            compute_kinetics(
                subset["S1"].values, subset["Rate"].values, condition=s2_val
            )
            for s2_val in df["S2"].unique()
            for subset in [df[df["S2"] == s2_val]]
        ]
    return [compute_kinetics(df["S1"].values, df["Rate"].values)]


def main() -> None:
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        print(f"No filename given, using '{DEFAULT_DATA_FILE}'")
        print(
            "To use own data give the filename as an argument: python assignment_1.py 'filename'"
        )
        filename = DEFAULT_DATA_FILE

    df = load_data(filename)

    # Sub-question 1: Mechanism identification
    if "S2" in df.columns:
        mech = identify_mechanism(df)
        print_mechanism(mech)

    # Sub-question 2: Diagnostic plots and kinetic parameters per S2 condition
    results = analyze(df)
    print_results(results, multi_condition=len(results) > 1)
    plot_michaelis_menten(df, results)
    plot_lineweaver_burk(df, results)

    # Sub-question 3: Batch reactor time (using S2-saturating condition)
    if "S2" in df.columns:
        saturating = [r for r in results if r.condition == df["S2"].max()]
        if saturating:
            batch = compute_batch_reactor_time(saturating[0].km, saturating[0].vmax)
            print_batch_reactor_time(batch)
    else:
        batch = compute_batch_reactor_time(results[0].km, results[0].vmax)
        print_batch_reactor_time(batch)



####### Part 2 ########


if __name__ == "__main__":
    main()
