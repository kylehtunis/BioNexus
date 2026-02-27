"""
Model Comparison Module for Gene Regulatory Network
Compares CNM, PWL, and Discrete models from Polynikis et al. (2009)

Generates side-by-side comparisons and overlay plots.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Import model modules
from cnm_model import CNMParameters, hill_activation, hill_inhibition, solve_cnm
from discrete_model import DiscreteParameters, solve_discrete
from pwl_model import PWLParameters, solve_pwl, step_activation, step_inhibition

# ===============================
# PLOT CONFIGURATION
# ===============================
plt.rcParams.update(
    {
        "figure.figsize": (8, 6),
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "savefig.dpi": 300,
        "savefig.format": "eps",
    }
)

# Model colors
COLOR_CNM = "#1f77b4"  # Blue
COLOR_PWL = "#ff7f0e"  # Orange
COLOR_DISCRETE = "#2ca02c"  # Green


# ===============================
# SOLVE ALL MODELS
# ===============================
def solve_all_models(
    y0: List[float] = [0.8, 0.8, 0.8, 0.8],
    t_span: Tuple[float, float] = (0, 200),
    dt: float = 0.1,
):
    """
    Solve all three models with identical parameters and initial conditions.

    Note: Discrete model uses Paper Eq. 23 (2 variables, exact integration),
    so it only uses pA0, pB0 from y0.

    Returns:
        dict with keys 'cnm', 'pwl', 'discrete' containing solutions
    """
    # CNM (full 4-variable continuous, Hill functions)
    params_cnm = CNMParameters()
    sol_cnm = solve_cnm(params_cnm, y0=y0, t_span=t_span, n_points=5000)

    # PWL (full 4-variable continuous, step functions)
    params_pwl = PWLParameters()
    sol_pwl = solve_pwl(params_pwl, y0=y0, t_span=t_span, n_points=5000)

    # Discrete (2-variable, Paper Eq. 23)
    params_discrete = DiscreteParameters()
    t_discrete, y_discrete = solve_discrete(params_discrete, y0=y0, t_span=t_span)

    return {"cnm": sol_cnm, "pwl": sol_pwl, "discrete": (t_discrete, y_discrete)}


# ===============================
# COMPARISON PLOTS
# ===============================
def plot_protein_timeseries_comparison(solutions: dict, save_path: str = None):
    """
    Overlay protein time series from all three models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sol_cnm = solutions["cnm"]
    sol_pwl = solutions["pwl"]
    t_discrete, y_discrete = solutions["discrete"]

    # Protein A
    axes[0].plot(
        sol_cnm.t, sol_cnm.y[2], color=COLOR_CNM, linewidth=2, label="CNM (Hill)"
    )
    axes[0].plot(
        sol_pwl.t,
        sol_pwl.y[2],
        color=COLOR_PWL,
        linewidth=2,
        linestyle="--",
        label="PWL (Step)",
    )
    axes[0].plot(
        t_discrete,
        y_discrete[2],
        color=COLOR_DISCRETE,
        linewidth=2,
        linestyle=":",
        label="Discrete",
    )
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Protein A Concentration [M]")
    axes[0].set_title("Protein A (GUARDIAN) Comparison")
    axes[0].legend()
    axes[0].set_xlim(0, 200)

    # Protein B
    axes[1].plot(
        sol_cnm.t, sol_cnm.y[3], color=COLOR_CNM, linewidth=2, label="CNM (Hill)"
    )
    axes[1].plot(
        sol_pwl.t,
        sol_pwl.y[3],
        color=COLOR_PWL,
        linewidth=2,
        linestyle="--",
        label="PWL (Step)",
    )
    axes[1].plot(
        t_discrete,
        y_discrete[3],
        color=COLOR_DISCRETE,
        linewidth=2,
        linestyle=":",
        label="Discrete",
    )
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Protein B Concentration [M]")
    axes[1].set_title("Protein B (PROLIFERATOR) Comparison")
    axes[1].legend()
    axes[1].set_xlim(0, 200)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def plot_mrna_timeseries_comparison(solutions: dict, save_path: str = None):
    """
    Overlay mRNA time series from all three models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sol_cnm = solutions["cnm"]
    sol_pwl = solutions["pwl"]
    t_discrete, y_discrete = solutions["discrete"]

    # mRNA A
    axes[0].plot(
        sol_cnm.t, sol_cnm.y[0], color=COLOR_CNM, linewidth=2, label="CNM (Hill)"
    )
    axes[0].plot(
        sol_pwl.t,
        sol_pwl.y[0],
        color=COLOR_PWL,
        linewidth=2,
        linestyle="--",
        label="PWL (Step)",
    )
    axes[0].plot(
        t_discrete,
        y_discrete[0],
        color=COLOR_DISCRETE,
        linewidth=2,
        linestyle=":",
        label="Discrete",
    )
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("mRNA A Concentration [M]")
    axes[0].set_title("mRNA A Comparison")
    axes[0].legend()
    axes[0].set_xlim(0, 200)

    # mRNA B
    axes[1].plot(
        sol_cnm.t, sol_cnm.y[1], color=COLOR_CNM, linewidth=2, label="CNM (Hill)"
    )
    axes[1].plot(
        sol_pwl.t,
        sol_pwl.y[1],
        color=COLOR_PWL,
        linewidth=2,
        linestyle="--",
        label="PWL (Step)",
    )
    axes[1].plot(
        t_discrete,
        y_discrete[1],
        color=COLOR_DISCRETE,
        linewidth=2,
        linestyle=":",
        label="Discrete",
    )
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("mRNA B Concentration [M]")
    axes[1].set_title("mRNA B Comparison")
    axes[1].legend()
    axes[1].set_xlim(0, 200)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def plot_phase_portrait_comparison(solutions: dict, save_path: str = None):
    """
    Overlay phase portraits from all three models.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sol_cnm = solutions["cnm"]
    sol_pwl = solutions["pwl"]
    t_discrete, y_discrete = solutions["discrete"]

    # CNM trajectory
    ax.plot(
        sol_cnm.y[2], sol_cnm.y[3], color=COLOR_CNM, linewidth=2, label="CNM (Hill)"
    )
    ax.plot(sol_cnm.y[2][0], sol_cnm.y[3][0], "o", color=COLOR_CNM, markersize=10)
    ax.plot(sol_cnm.y[2][-1], sol_cnm.y[3][-1], "*", color=COLOR_CNM, markersize=15)

    # PWL trajectory
    ax.plot(
        sol_pwl.y[2],
        sol_pwl.y[3],
        color=COLOR_PWL,
        linewidth=2,
        linestyle="--",
        label="PWL (Step)",
    )
    ax.plot(sol_pwl.y[2][-1], sol_pwl.y[3][-1], "*", color=COLOR_PWL, markersize=15)

    # Discrete trajectory
    ax.plot(
        y_discrete[2],
        y_discrete[3],
        color=COLOR_DISCRETE,
        linewidth=2,
        linestyle=":",
        label="Discrete",
    )
    ax.plot(
        y_discrete[2][-1], y_discrete[3][-1], "*", color=COLOR_DISCRETE, markersize=15
    )

    # Threshold lines
    ax.axvline(x=0.21, color="gray", linestyle="--", alpha=0.5, label="$\\theta_A$")
    ax.axhline(y=0.21, color="gray", linestyle=":", alpha=0.5, label="$\\theta_B$")

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title("Phase Portrait Comparison: CNM vs PWL vs Discrete")
    ax.legend(loc="best")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_phase_portraits_side_by_side(solutions: dict, save_path: str = None):
    """
    Plot phase portraits side-by-side for comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sol_cnm = solutions["cnm"]
    sol_pwl = solutions["pwl"]
    t_discrete, y_discrete = solutions["discrete"]

    # CNM
    axes[0].plot(sol_cnm.y[2], sol_cnm.y[3], color=COLOR_CNM, linewidth=2)
    axes[0].plot(sol_cnm.y[2][0], sol_cnm.y[3][0], "go", markersize=10, label="Start")
    axes[0].plot(sol_cnm.y[2][-1], sol_cnm.y[3][-1], "r*", markersize=15, label="End")
    axes[0].axvline(x=0.21, color="gray", linestyle="--", alpha=0.5)
    axes[0].axhline(y=0.21, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Protein A [M]")
    axes[0].set_ylabel("Protein B [M]")
    axes[0].set_title("CNM (Hill Functions)")
    axes[0].legend()
    axes[0].set_xlim(0, 2.5)
    axes[0].set_ylim(0, 1.5)

    # PWL
    axes[1].plot(sol_pwl.y[2], sol_pwl.y[3], color=COLOR_PWL, linewidth=2)
    axes[1].plot(sol_pwl.y[2][0], sol_pwl.y[3][0], "go", markersize=10, label="Start")
    axes[1].plot(sol_pwl.y[2][-1], sol_pwl.y[3][-1], "r*", markersize=15, label="End")
    axes[1].axvline(x=0.21, color="gray", linestyle="--", alpha=0.5)
    axes[1].axhline(y=0.21, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Protein A [M]")
    axes[1].set_ylabel("Protein B [M]")
    axes[1].set_title("PWL (Step Functions)")
    axes[1].legend()
    axes[1].set_xlim(0, 2.5)
    axes[1].set_ylim(0, 1.5)

    # Discrete
    axes[2].plot(y_discrete[2], y_discrete[3], color=COLOR_DISCRETE, linewidth=2)
    axes[2].plot(y_discrete[2][0], y_discrete[3][0], "go", markersize=10, label="Start")
    axes[2].plot(y_discrete[2][-1], y_discrete[3][-1], "r*", markersize=15, label="End")
    axes[2].axvline(x=0.21, color="gray", linestyle="--", alpha=0.5)
    axes[2].axhline(y=0.21, color="gray", linestyle=":", alpha=0.5)
    axes[2].set_xlabel("Protein A [M]")
    axes[2].set_ylabel("Protein B [M]")
    axes[2].set_title("Discrete (Forward Euler)")
    axes[2].legend()
    axes[2].set_xlim(0, 2.5)
    axes[2].set_ylim(0, 1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def plot_transcription_functions_comparison(save_path: str = None):
    """
    Compare Hill functions with PWL step functions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    p = np.linspace(0, 1.5, 500)
    theta = 0.21

    # Different Hill coefficients
    for n in [1, 2, 3, 5, 10]:
        h_act = hill_activation(p, theta, n)
        axes[0].plot(p, h_act, label=f"n = {n}")

        h_inh = hill_inhibition(p, theta, n)
        axes[1].plot(p, h_inh, label=f"n = {n}")

    # PWL step function
    s_act = step_activation(p, theta, steepness=100)
    axes[0].plot(p, s_act, "k--", linewidth=2, label="PWL (n→∞)")

    s_inh = step_inhibition(p, theta, steepness=100)
    axes[1].plot(p, s_inh, "k--", linewidth=2, label="PWL (n→∞)")

    # Threshold line
    axes[0].axvline(x=theta, color="gray", linestyle=":", alpha=0.5)
    axes[1].axvline(x=theta, color="gray", linestyle=":", alpha=0.5)

    axes[0].set_xlabel("Protein Concentration [M]")
    axes[0].set_ylabel("Activation Level")
    axes[0].set_title("Activation: Hill Coefficient Effect")
    axes[0].legend()
    axes[0].set_xlim(0, 1.5)
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].set_xlabel("Protein Concentration [M]")
    axes[1].set_ylabel("Inhibition Level")
    axes[1].set_title("Inhibition: Hill Coefficient Effect")
    axes[1].legend()
    axes[1].set_xlim(0, 1.5)
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def print_steady_state_comparison(solutions: dict):
    """
    Print comparison table of steady-state values.
    """
    sol_cnm = solutions["cnm"]
    sol_pwl = solutions["pwl"]
    t_discrete, y_discrete = solutions["discrete"]

    print("\n" + "=" * 70)
    print("STEADY-STATE COMPARISON")
    print("=" * 70)
    print(f"{'Variable':<15} {'CNM (Hill)':<15} {'PWL (Step)':<15} {'Discrete':<15}")
    print("-" * 70)
    print(
        f"{'mRNA A [M]':<15} {sol_cnm.y[0][-1]:<15.4f} {sol_pwl.y[0][-1]:<15.4f} {y_discrete[0][-1]:<15.4f}"
    )
    print(
        f"{'mRNA B [M]':<15} {sol_cnm.y[1][-1]:<15.4f} {sol_pwl.y[1][-1]:<15.4f} {y_discrete[1][-1]:<15.4f}"
    )
    print(
        f"{'Protein A [M]':<15} {sol_cnm.y[2][-1]:<15.4f} {sol_pwl.y[2][-1]:<15.4f} {y_discrete[2][-1]:<15.4f}"
    )
    print(
        f"{'Protein B [M]':<15} {sol_cnm.y[3][-1]:<15.4f} {sol_pwl.y[3][-1]:<15.4f} {y_discrete[3][-1]:<15.4f}"
    )
    print("=" * 70)

    # Interpretation
    print("\nINTERPRETATION:")
    print("-" * 70)

    pA_cnm, pB_cnm = sol_cnm.y[2][-1], sol_cnm.y[3][-1]
    pA_pwl, pB_pwl = sol_pwl.y[2][-1], sol_pwl.y[3][-1]
    pA_discrete, pB_discrete = y_discrete[2][-1], y_discrete[3][-1]

    print(
        f"CNM:      Protein A (GUARDIAN) {'dominates' if pA_cnm > pB_cnm else 'suppressed'}"
    )
    print(f"          → Tumor suppressor {'active' if pA_cnm > 0.5 else 'inactive'}")
    print(
        f"PWL:      Protein A (GUARDIAN) {'dominates' if pA_pwl > pB_pwl else 'suppressed'}"
    )
    print(f"          → Tumor suppressor {'active' if pA_pwl > 0.5 else 'inactive'}")
    print(
        f"Discrete: Protein A (GUARDIAN) {'dominates' if pA_discrete > pB_discrete else 'suppressed'}"
    )
    print(
        f"          → Tumor suppressor {'active' if pA_discrete > 0.5 else 'inactive'}"
    )
    print("=" * 70 + "\n")


def plot_model_comparison_summary(solutions: dict, save_path: str = None):
    """
    Create a comprehensive summary figure comparing all models.
    """
    fig = plt.figure(figsize=(16, 12))

    sol_cnm = solutions["cnm"]
    sol_pwl = solutions["pwl"]
    t_discrete, y_discrete = solutions["discrete"]

    # 1. Protein A time series (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(sol_cnm.t, sol_cnm.y[2], color=COLOR_CNM, linewidth=2, label="CNM")
    ax1.plot(
        sol_pwl.t,
        sol_pwl.y[2],
        color=COLOR_PWL,
        linewidth=2,
        linestyle="--",
        label="PWL",
    )
    ax1.plot(
        t_discrete,
        y_discrete[2],
        color=COLOR_DISCRETE,
        linewidth=2,
        linestyle=":",
        label="Discrete",
    )
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Concentration [M]")
    ax1.set_title("Protein A (GUARDIAN)")
    ax1.legend()
    ax1.set_xlim(0, 50)

    # 2. Protein B time series (top middle)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(sol_cnm.t, sol_cnm.y[3], color=COLOR_CNM, linewidth=2, label="CNM")
    ax2.plot(
        sol_pwl.t,
        sol_pwl.y[3],
        color=COLOR_PWL,
        linewidth=2,
        linestyle="--",
        label="PWL",
    )
    ax2.plot(
        t_discrete,
        y_discrete[3],
        color=COLOR_DISCRETE,
        linewidth=2,
        linestyle=":",
        label="Discrete",
    )
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Concentration [M]")
    ax2.set_title("Protein B (PROLIFERATOR)")
    ax2.legend()
    ax2.set_xlim(0, 50)

    # 3. Phase portrait overlay (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(sol_cnm.y[2], sol_cnm.y[3], color=COLOR_CNM, linewidth=2, label="CNM")
    ax3.plot(
        sol_pwl.y[2],
        sol_pwl.y[3],
        color=COLOR_PWL,
        linewidth=2,
        linestyle="--",
        label="PWL",
    )
    ax3.plot(
        y_discrete[2],
        y_discrete[3],
        color=COLOR_DISCRETE,
        linewidth=2,
        linestyle=":",
        label="Discrete",
    )
    ax3.axvline(x=0.21, color="gray", linestyle="--", alpha=0.3)
    ax3.axhline(y=0.21, color="gray", linestyle="--", alpha=0.3)
    ax3.set_xlabel("Protein A [M]")
    ax3.set_ylabel("Protein B [M]")
    ax3.set_title("Phase Portrait Overlay")
    ax3.legend()
    ax3.set_xlim(0, 2.5)
    ax3.set_ylim(0, 1.0)

    # 4. mRNA A time series (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(sol_cnm.t, sol_cnm.y[0], color=COLOR_CNM, linewidth=2, label="CNM")
    ax4.plot(
        sol_pwl.t,
        sol_pwl.y[0],
        color=COLOR_PWL,
        linewidth=2,
        linestyle="--",
        label="PWL",
    )
    ax4.plot(
        t_discrete,
        y_discrete[0],
        color=COLOR_DISCRETE,
        linewidth=2,
        linestyle=":",
        label="Discrete",
    )
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Concentration [M]")
    ax4.set_title("mRNA A")
    ax4.legend()
    ax4.set_xlim(0, 50)

    # 5. mRNA B time series (bottom middle)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(sol_cnm.t, sol_cnm.y[1], color=COLOR_CNM, linewidth=2, label="CNM")
    ax5.plot(
        sol_pwl.t,
        sol_pwl.y[1],
        color=COLOR_PWL,
        linewidth=2,
        linestyle="--",
        label="PWL",
    )
    ax5.plot(
        t_discrete,
        y_discrete[1],
        color=COLOR_DISCRETE,
        linewidth=2,
        linestyle=":",
        label="Discrete",
    )
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Concentration [M]")
    ax5.set_title("mRNA B")
    ax5.legend()
    ax5.set_xlim(0, 50)

    # 6. Steady-state bar chart (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    variables = ["$p_A$", "$p_B$", "$r_A$", "$r_B$"]
    x = np.arange(len(variables))
    width = 0.25

    cnm_values = [
        sol_cnm.y[2][-1],
        sol_cnm.y[3][-1],
        sol_cnm.y[0][-1],
        sol_cnm.y[1][-1],
    ]
    pwl_values = [
        sol_pwl.y[2][-1],
        sol_pwl.y[3][-1],
        sol_pwl.y[0][-1],
        sol_pwl.y[1][-1],
    ]
    discrete_values = [
        y_discrete[2][-1],
        y_discrete[3][-1],
        y_discrete[0][-1],
        y_discrete[1][-1],
    ]

    ax6.bar(x - width, cnm_values, width, label="CNM", color=COLOR_CNM)
    ax6.bar(x, pwl_values, width, label="PWL", color=COLOR_PWL)
    ax6.bar(x + width, discrete_values, width, label="Discrete", color=COLOR_DISCRETE)
    ax6.set_xlabel("Variable")
    ax6.set_ylabel("Steady-State Value [M]")
    ax6.set_title("Steady-State Comparison")
    ax6.set_xticks(x)
    ax6.set_xticklabels(variables)
    ax6.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ===============================
# MAIN EXECUTION
# ===============================
def main():
    """Run model comparison and generate all comparison plots."""

    print("\n" + "=" * 60)
    print("MODEL COMPARISON: CNM vs PWL vs DISCRETE")
    print("Gene Regulatory Network: Activator-Inhibitor")
    print("=" * 60)

    # Initial conditions
    y0 = [0.8, 0.8, 0.8, 0.8]

    # Solve all models
    print("\nSolving all models...")
    solutions = solve_all_models(y0=y0, t_span=(0, 50), dt=0.1)

    # Print comparison table
    print_steady_state_comparison(solutions)

    # Generate comparison plots
    print("Generating comparison plots...")

    # 1. Protein time series comparison
    plot_protein_timeseries_comparison(
        solutions, save_path="output/comparison_protein_timeseries.eps"
    )

    # 2. mRNA time series comparison
    plot_mrna_timeseries_comparison(
        solutions, save_path="output/comparison_mrna_timeseries.eps"
    )

    # 3. Phase portrait overlay
    plot_phase_portrait_comparison(
        solutions, save_path="output/comparison_phase_portrait.eps"
    )

    # 4. Phase portraits side-by-side
    plot_phase_portraits_side_by_side(
        solutions, save_path="output/comparison_phase_side_by_side.eps"
    )

    # 5. Transcription functions
    plot_transcription_functions_comparison(
        save_path="output/comparison_transcription_functions.eps"
    )

    # 6. Summary figure
    plot_model_comparison_summary(solutions, save_path="output/comparison_summary.eps")

    print("\nAll comparison plots generated successfully!")
    print("Files saved in current directory with .eps format")

    # Show all plots
    plt.show()

    return solutions


if __name__ == "__main__":
    solutions = main()
