"""
Complete Nonlinear Model (CNM) for Gene Regulatory Network
Based on the activator-inhibitor network from Polynikis et al. (2009)

Gene A (GUARDIAN): Tumor-suppressor - activated by Protein B
Gene B (PROLIFERATOR): Oncogenic - inhibited by Protein A

Mechanism I: Transcriptional regulation (ODE model)
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

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

# Standard colors
COLOR_PA = "#1f77b4"  # Blue - Protein A
COLOR_PB = "#ff7f0e"  # Orange - Protein B
COLOR_RA = "#2ca02c"  # Green - mRNA A
COLOR_RB = "#d62728"  # Red - mRNA B


# ===============================
# MODEL PARAMETERS (Table 5 - ODE)
# ===============================
class CNMParameters:
    """Parameters for the Complete Nonlinear Model (Mechanism I)"""

    def __init__(self):
        # Transcription rates [s^-1]
        self.mA = 2.35  # max transcription rate of Gene A
        self.mB = 2.35  # max transcription rate of Gene B

        # mRNA degradation rates [s^-1]
        self.gammaA = 1.0  # mRNA A degradation rate
        self.gammaB = 1.0  # mRNA B degradation rate

        # Translation rates [s^-1]
        self.kPA = 1.0  # translation rate of Protein A
        self.kPB = 1.0  # translation rate of Protein B

        # Expression thresholds [M]
        self.thetaA = 0.21  # threshold for Protein A binding
        self.thetaB = 0.21  # threshold for Protein B binding

        # Hill coefficients (cooperativity)
        self.nA = 3  # Hill coefficient for Protein A
        self.nB = 3  # Hill coefficient for Protein B

        # Protein degradation rates [s^-1]
        self.deltaPA = 1.0  # degradation rate of Protein A
        self.deltaPB = 1.0  # degradation rate of Protein B


# ===============================
# HILL FUNCTIONS
# ===============================
def hill_activation(p: np.ndarray, theta: float, n: int) -> np.ndarray:
    """
    Hill function for activation (increasing sigmoidal).
    h+(p) = p^n / (p^n + theta^n)

    Args:
        p: Protein concentration [M]
        theta: Expression threshold [M]
        n: Hill coefficient (cooperativity)

    Returns:
        Activation level [dimensionless, 0-1]
    """
    return np.power(p, n) / (np.power(p, n) + np.power(theta, n))


def hill_inhibition(p: np.ndarray, theta: float, n: int) -> np.ndarray:
    """
    Hill function for inhibition (decreasing sigmoidal).
    h-(p) = theta^n / (p^n + theta^n)

    Args:
        p: Protein concentration [M]
        theta: Expression threshold [M]
        n: Hill coefficient (cooperativity)

    Returns:
        Inhibition level [dimensionless, 0-1]
    """
    return np.power(theta, n) / (np.power(p, n) + np.power(theta, n))


# ===============================
# CNM ODE SYSTEM
# ===============================
def cnm_ode(t: float, y: np.ndarray, params: CNMParameters) -> List[float]:
    """
    Complete Nonlinear Model ODE system.

    State variables:
        y[0] = rA: mRNA A concentration [M]
        y[1] = rB: mRNA B concentration [M]
        y[2] = pA: Protein A concentration [M]
        y[3] = pB: Protein B concentration [M]

    ODEs:
        drA/dt = mA * h+(pB, θB, nB) - γA * rA
        drB/dt = mB * h-(pA, θA, nA) - γB * rB
        dpA/dt = kPA * rA - δPA * pA
        dpB/dt = kPB * rB - δPB * pB

    Args:
        t: Time [s]
        y: State vector [rA, rB, pA, pB]
        params: Model parameters

    Returns:
        Derivatives [drA/dt, drB/dt, dpA/dt, dpB/dt]
    """
    rA, rB, pA, pB = y

    # Gene A is activated by Protein B
    drA = params.mA * hill_activation(pB, params.thetaB, params.nB) - params.gammaA * rA

    # Gene B is inhibited by Protein A
    drB = params.mB * hill_inhibition(pA, params.thetaA, params.nA) - params.gammaB * rB

    # Protein translation and degradation
    dpA = params.kPA * rA - params.deltaPA * pA
    dpB = params.kPB * rB - params.deltaPB * pB

    return [drA, drB, dpA, dpB]


def solve_cnm(
    params: CNMParameters,
    y0: List[float] = [0.8, 0.8, 0.8, 0.8],
    t_span: Tuple[float, float] = (0, 50),
    n_points: int = 2000,
):
    """
    Solve the CNM ODE system.

    Args:
        params: Model parameters
        y0: Initial conditions [rA0, rB0, pA0, pB0] in [M]
        t_span: Time interval [s]
        n_points: Number of evaluation points

    Returns:
        scipy OdeResult object
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        lambda t, y: cnm_ode(t, y, params),
        t_span,
        y0,
        t_eval=t_eval,
        method="RK45",
        dense_output=True,
    )
    return sol


# ===============================
# PLOTTING FUNCTIONS
# ===============================
def plot_time_series_proteins(sol, save_path: str = None):
    """
    Plot protein concentrations vs time.

    Args:
        sol: ODE solution object
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots()

    ax.plot(sol.t, sol.y[2], color=COLOR_PA, label="Protein A (GUARDIAN)")
    ax.plot(sol.t, sol.y[3], color=COLOR_PB, label="Protein B (PROLIFERATOR)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Protein Concentration [M]")
    ax.set_title("CNM: Protein Time Series")
    ax.legend(loc="best")
    ax.set_xlim(sol.t[0], sol.t[-1])
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_time_series_mrna(sol, save_path: str = None):
    """
    Plot mRNA concentrations vs time.

    Args:
        sol: ODE solution object
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots()

    ax.plot(sol.t, sol.y[0], color=COLOR_RA, label="mRNA A")
    ax.plot(sol.t, sol.y[1], color=COLOR_RB, label="mRNA B")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("mRNA Concentration [M]")
    ax.set_title("CNM: mRNA Time Series")
    ax.legend(loc="best")
    ax.set_xlim(sol.t[0], sol.t[-1])
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_time_series_all(sol, save_path: str = None):
    """
    Plot all state variables vs time in subplots.

    Args:
        sol: ODE solution object
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # mRNA A
    axes[0, 0].plot(sol.t, sol.y[0], color=COLOR_RA)
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Concentration [M]")
    axes[0, 0].set_title("mRNA A ($r_A$)")
    axes[0, 0].set_xlim(sol.t[0], sol.t[-1])

    # mRNA B
    axes[0, 1].plot(sol.t, sol.y[1], color=COLOR_RB)
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Concentration [M]")
    axes[0, 1].set_title("mRNA B ($r_B$)")
    axes[0, 1].set_xlim(sol.t[0], sol.t[-1])

    # Protein A
    axes[1, 0].plot(sol.t, sol.y[2], color=COLOR_PA)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Concentration [M]")
    axes[1, 0].set_title("Protein A ($p_A$)")
    axes[1, 0].set_xlim(sol.t[0], sol.t[-1])

    # Protein B
    axes[1, 1].plot(sol.t, sol.y[3], color=COLOR_PB)
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Concentration [M]")
    axes[1, 1].set_title("Protein B ($p_B$)")
    axes[1, 1].set_xlim(sol.t[0], sol.t[-1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def plot_phase_portrait_proteins(sol, params: CNMParameters, save_path: str = None):
    """
    Plot phase portrait in Protein A - Protein B plane with nullclines.

    Args:
        sol: ODE solution object
        params: Model parameters
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots()

    # Plot trajectory
    ax.plot(sol.y[2], sol.y[3], color="#9467bd", linewidth=1.5, label="Trajectory")

    # Mark start and end points
    ax.plot(sol.y[2][0], sol.y[3][0], "go", markersize=10, label="Start", zorder=5)
    ax.plot(
        sol.y[2][-1],
        sol.y[3][-1],
        "r*",
        markersize=15,
        label="End (Steady State)",
        zorder=5,
    )

    # Direction arrows along trajectory
    n_arrows = 10
    idx = np.linspace(0, len(sol.t) - 2, n_arrows, dtype=int)
    for i in idx:
        dx = sol.y[2][i + 1] - sol.y[2][i]
        dy = sol.y[3][i + 1] - sol.y[3][i]
        ax.annotate(
            "",
            xy=(sol.y[2][i + 1], sol.y[3][i + 1]),
            xytext=(sol.y[2][i], sol.y[3][i]),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        )

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title("CNM: Phase Portrait (Protein Space)")
    ax.legend(loc="best")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_vector_field_proteins(
    params: CNMParameters,
    p_range: Tuple[float, float] = (0, 3),
    n_grid: int = 20,
    save_path: str = None,
):
    """
    Plot vector field (stream plot) in protein phase space.

    Under quasi-steady-state mRNA assumption:
        dpA/dt ≈ (kPA/γA) * mA * h+(pB) - δPA * pA
        dpB/dt ≈ (kPB/γB) * mB * h-(pA) - δPB * pB

    Args:
        params: Model parameters
        p_range: Range for protein concentrations
        n_grid: Grid resolution
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots()

    # Create meshgrid
    pA = np.linspace(p_range[0], p_range[1], n_grid)
    pB = np.linspace(p_range[0], p_range[1], n_grid)
    PA, PB = np.meshgrid(pA, pB)

    # Quasi-steady-state approximation for mRNA
    # At steady state: rA = (mA/γA) * h+(pB), rB = (mB/γB) * h-(pA)
    RA_ss = (params.mA / params.gammaA) * hill_activation(PB, params.thetaB, params.nB)
    RB_ss = (params.mB / params.gammaB) * hill_inhibition(PA, params.thetaA, params.nA)

    # Protein dynamics
    dpA_dt = params.kPA * RA_ss - params.deltaPA * PA
    dpB_dt = params.kPB * RB_ss - params.deltaPB * PB

    # Normalize for visualization
    magnitude = np.sqrt(dpA_dt**2 + dpB_dt**2)
    magnitude[magnitude == 0] = 1

    # Stream plot
    strm = ax.streamplot(
        PA,
        PB,
        dpA_dt,
        dpB_dt,
        color=magnitude,
        cmap="viridis",
        density=1.5,
        linewidth=1,
        arrowsize=1.5,
    )

    # Nullclines (where dpA/dt = 0 and dpB/dt = 0)
    # pA nullcline: pA = (kPA/δPA) * (mA/γA) * h+(pB)
    pB_null = np.linspace(0.01, p_range[1], 200)
    pA_nullcline = (
        params.kPA * params.mA / (params.deltaPA * params.gammaA)
    ) * hill_activation(pB_null, params.thetaB, params.nB)

    # pB nullcline: pB = (kPB/δPB) * (mB/γB) * h-(pA)
    pA_null = np.linspace(0.01, p_range[1], 200)
    pB_nullcline = (
        params.kPB * params.mB / (params.deltaPB * params.gammaB)
    ) * hill_inhibition(pA_null, params.thetaA, params.nA)

    ax.plot(pA_nullcline, pB_null, "b--", linewidth=2, label="$dp_A/dt = 0$")
    ax.plot(pA_null, pB_nullcline, "r--", linewidth=2, label="$dp_B/dt = 0$")

    # Find approximate equilibrium
    eq_pA = (
        (params.kPA * params.mA) / (params.deltaPA * params.gammaA) * 0.5
    )  # approximation
    eq_pB = (params.kPB * params.mB) / (params.deltaPB * params.gammaB) * 0.5
    ax.plot(eq_pA, eq_pB, "ko", markersize=10, label="Equilibrium (approx)")

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title("CNM: Vector Field with Nullclines\n(Quasi-Steady-State mRNA)")
    ax.legend(loc="upper right")
    ax.set_xlim(p_range)
    ax.set_ylim(p_range)

    plt.colorbar(strm.lines, label="Vector Magnitude")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_multiple_trajectories(
    params: CNMParameters,
    initial_conditions: List[List[float]] = None,
    t_span: Tuple[float, float] = (0, 50),
    save_path: str = None,
):
    """
    Plot multiple trajectories from different initial conditions.

    Args:
        params: Model parameters
        initial_conditions: List of [rA0, rB0, pA0, pB0]
        t_span: Time interval
        save_path: Path to save figure (optional)
    """
    if initial_conditions is None:
        initial_conditions = [
            [0.1, 0.1, 0.1, 0.1],
            [0.8, 0.8, 0.8, 0.8],
            [1.5, 0.5, 1.5, 0.5],
            [0.5, 1.5, 0.5, 1.5],
            [2.0, 2.0, 2.0, 2.0],
            [0.2, 1.0, 0.2, 1.0],
        ]

    fig, ax = plt.subplots()

    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions)))

    for i, y0 in enumerate(initial_conditions):
        sol = solve_cnm(params, y0=y0, t_span=t_span)
        ax.plot(
            sol.y[2],
            sol.y[3],
            color=colors[i],
            linewidth=1.5,
            label=f"IC: $p_A$={y0[2]:.1f}, $p_B$={y0[3]:.1f}",
        )
        ax.plot(sol.y[2][0], sol.y[3][0], "o", color=colors[i], markersize=8)
        ax.plot(sol.y[2][-1], sol.y[3][-1], "*", color=colors[i], markersize=12)

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title("CNM: Multiple Trajectories from Different Initial Conditions")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_hill_functions(params: CNMParameters, save_path: str = None):
    """
    Plot the Hill activation and inhibition functions.

    Args:
        params: Model parameters
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    p = np.linspace(0, 1.5, 200)

    # Activation
    h_act = hill_activation(p, params.thetaA, params.nA)
    axes[0].plot(p, h_act, color=COLOR_PA, linewidth=2)
    axes[0].axvline(
        x=params.thetaA,
        color="gray",
        linestyle="--",
        label=f"$\\theta_A = {params.thetaA}$ M",
    )
    axes[0].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Protein Concentration [M]")
    axes[0].set_ylabel("Activation Level")
    axes[0].set_title(f"Hill Activation Function ($n = {params.nA}$)")
    axes[0].legend()
    axes[0].set_xlim(0, 1.5)
    axes[0].set_ylim(0, 1.05)

    # Inhibition
    h_inh = hill_inhibition(p, params.thetaB, params.nB)
    axes[1].plot(p, h_inh, color=COLOR_PB, linewidth=2)
    axes[1].axvline(
        x=params.thetaB,
        color="gray",
        linestyle="--",
        label=f"$\\theta_B = {params.thetaB}$ M",
    )
    axes[1].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Protein Concentration [M]")
    axes[1].set_ylabel("Inhibition Level")
    axes[1].set_title(f"Hill Inhibition Function ($n = {params.nB}$)")
    axes[1].legend()
    axes[1].set_xlim(0, 1.5)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def print_steady_state(sol):
    """Print the approximate steady-state values."""
    print("\n" + "=" * 50)
    print("STEADY-STATE VALUES (approximate)")
    print("=" * 50)
    print(f"mRNA A (rA):     {sol.y[0][-1]:.4f} M")
    print(f"mRNA B (rB):     {sol.y[1][-1]:.4f} M")
    print(f"Protein A (pA):  {sol.y[2][-1]:.4f} M")
    print(f"Protein B (pB):  {sol.y[3][-1]:.4f} M")
    print("=" * 50 + "\n")


# ===============================
# MAIN EXECUTION
# ===============================
def main():
    """Run the Complete Nonlinear Model simulation and generate plots."""

    print("\n" + "=" * 60)
    print("COMPLETE NONLINEAR MODEL (CNM)")
    print("Gene Regulatory Network: Activator-Inhibitor")
    print("=" * 60)

    # Initialize parameters
    params = CNMParameters()

    # Initial conditions [rA, rB, pA, pB] in M
    y0 = [0.8, 0.8, 0.8, 0.8]

    # Solve the system (t=200 to capture oscillatory dynamics)
    print("\nSolving CNM ODE system...")
    sol = solve_cnm(params, y0=y0, t_span=(0, 200), n_points=5000)

    # Print steady-state values
    print_steady_state(sol)

    # Check for oscillations
    pA_late = sol.y[2][sol.t > 150]
    pA_range = np.max(pA_late) - np.min(pA_late)
    if pA_range > 0.01:
        print(f"⚠️  System is OSCILLATING (pA amplitude: {pA_range:.4f} M)")
        print("   This is a limit cycle predicted by the full 4D CNM")
        print("   (see Polynikis et al. 2009, Section 5)")
    else:
        print("✅ System has converged to stable equilibrium")

    # Generate all plots
    print("Generating plots...")

    # 1. Hill functions
    plot_hill_functions(params, save_path="output/cnm_hill_functions.eps")

    # 2. Time series - Proteins only
    plot_time_series_proteins(sol, save_path="output/cnm_protein_timeseries.eps")

    # 3. Time series - mRNA only
    plot_time_series_mrna(sol, save_path="output/cnm_mrna_timeseries.eps")

    # 4. Time series - All variables
    plot_time_series_all(sol, save_path="output/cnm_all_timeseries.eps")

    # 5. Phase portrait
    plot_phase_portrait_proteins(sol, params, save_path="output/cnm_phase_portrait.eps")

    # 6. Vector field with nullclines
    plot_vector_field_proteins(params, save_path="output/cnm_vector_field.eps")

    # 7. Multiple trajectories
    plot_multiple_trajectories(params, save_path="output/cnm_multiple_trajectories.eps")

    print("\nAll plots generated successfully!")
    print("Files saved in current directory with .eps format")

    # Show all plots
    plt.show()

    return sol, params


if __name__ == "__main__":
    sol, params = main()
