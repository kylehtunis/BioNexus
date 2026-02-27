"""
Piecewise-Linear (PWL) Model for Gene Regulatory Network
Based on the activator-inhibitor network from Polynikis et al. (2009)

Replaces Hill functions with step functions (limit as n → ∞):
    s+(p, θ) = 0 if p < θ, 1 if p > θ  (activation)
    s-(p, θ) = 1 if p < θ, 0 if p > θ  (inhibition)

Gene A (GUARDIAN): Tumor-suppressor - activated by Protein B
Gene B (PROLIFERATOR): Oncogenic - inhibited by Protein A
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

# Standard colors (same as CNM for consistency)
COLOR_PA = "#1f77b4"  # Blue - Protein A
COLOR_PB = "#ff7f0e"  # Orange - Protein B
COLOR_RA = "#2ca02c"  # Green - mRNA A
COLOR_RB = "#d62728"  # Red - mRNA B


# ===============================
# MODEL PARAMETERS
# ===============================
class PWLParameters:
    """Parameters for the Piecewise-Linear Model"""

    def __init__(self):
        # Transcription rates [s^-1]
        self.mA = 2.35
        self.mB = 2.35

        # mRNA degradation rates [s^-1]
        self.gammaA = 1.0
        self.gammaB = 1.0

        # Translation rates [s^-1]
        self.kPA = 1.0
        self.kPB = 1.0

        # Expression thresholds [M]
        self.thetaA = 0.21
        self.thetaB = 0.21

        # Protein degradation rates [s^-1]
        self.deltaPA = 1.0
        self.deltaPB = 1.0


# ===============================
# PWL STEP FUNCTIONS
# ===============================
def step_activation(p: np.ndarray, theta: float, steepness: float = 100) -> np.ndarray:
    """
    Step function for activation (PWL approximation of Hill as n→∞).
    Uses smooth approximation to avoid numerical issues.

    s+(p, θ) ≈ 1 / (1 + exp(-steepness * (p - θ)))

    Args:
        p: Protein concentration [M]
        theta: Expression threshold [M]
        steepness: Steepness parameter (higher = sharper step)

    Returns:
        Activation level [dimensionless, 0-1]
    """
    return 1.0 / (1.0 + np.exp(-steepness * (p - theta)))


def step_inhibition(p: np.ndarray, theta: float, steepness: float = 100) -> np.ndarray:
    """
    Step function for inhibition (PWL approximation of Hill as n→∞).

    s-(p, θ) = 1 - s+(p, θ)

    Args:
        p: Protein concentration [M]
        theta: Expression threshold [M]
        steepness: Steepness parameter (higher = sharper step)

    Returns:
        Inhibition level [dimensionless, 0-1]
    """
    return 1.0 - step_activation(p, theta, steepness)


def step_activation_hard(p: float, theta: float) -> float:
    """Hard step function (discontinuous) for theoretical analysis."""
    if p < theta:
        return 0.0
    elif p > theta:
        return 1.0
    else:
        return 0.5  # Filippov convention at threshold


def step_inhibition_hard(p: float, theta: float) -> float:
    """Hard step function for inhibition."""
    return 1.0 - step_activation_hard(p, theta)


# ===============================
# PWL ODE SYSTEM
# ===============================
def pwl_ode(
    t: float, y: np.ndarray, params: PWLParameters, steepness: float = 100
) -> List[float]:
    """
    Piecewise-Linear Model ODE system.

    Same structure as CNM but with step functions instead of Hill functions.

    State variables:
        y[0] = rA: mRNA A concentration [M]
        y[1] = rB: mRNA B concentration [M]
        y[2] = pA: Protein A concentration [M]
        y[3] = pB: Protein B concentration [M]

    ODEs:
        drA/dt = mA * s+(pB, θB) - γA * rA
        drB/dt = mB * s-(pA, θA) - γB * rB
        dpA/dt = kPA * rA - δPA * pA
        dpB/dt = kPB * rB - δPB * pB
    """
    rA, rB, pA, pB = y

    # Gene A is activated by Protein B (step function)
    drA = params.mA * step_activation(pB, params.thetaB, steepness) - params.gammaA * rA

    # Gene B is inhibited by Protein A (step function)
    drB = params.mB * step_inhibition(pA, params.thetaA, steepness) - params.gammaB * rB

    # Protein translation and degradation (same as CNM)
    dpA = params.kPA * rA - params.deltaPA * pA
    dpB = params.kPB * rB - params.deltaPB * pB

    return [drA, drB, dpA, dpB]


def solve_pwl(
    params: PWLParameters,
    y0: List[float] = [0.8, 0.8, 0.8, 0.8],
    t_span: Tuple[float, float] = (0, 50),
    n_points: int = 2000,
    steepness: float = 100,
):
    """Solve the PWL ODE system."""
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        lambda t, y: pwl_ode(t, y, params, steepness),
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
def plot_step_functions(params: PWLParameters, save_path: str = None):
    """Plot PWL step functions compared to Hill functions."""
    from cnm_model import hill_activation, hill_inhibition

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    p = np.linspace(0, 1.5, 500)

    # Activation comparison
    h_act = hill_activation(p, params.thetaA, 3)
    s_act = step_activation(p, params.thetaA, steepness=100)

    axes[0].plot(p, h_act, "b-", linewidth=2, label="Hill (n=3)")
    axes[0].plot(p, s_act, "r--", linewidth=2, label="PWL Step")
    axes[0].axvline(
        x=params.thetaA, color="gray", linestyle=":", label=f"θ = {params.thetaA} M"
    )
    axes[0].set_xlabel("Protein Concentration [M]")
    axes[0].set_ylabel("Activation Level")
    axes[0].set_title("Activation: Hill vs PWL Step")
    axes[0].legend()
    axes[0].set_xlim(0, 1.5)
    axes[0].set_ylim(-0.05, 1.05)

    # Inhibition comparison
    h_inh = hill_inhibition(p, params.thetaB, 3)
    s_inh = step_inhibition(p, params.thetaB, steepness=100)

    axes[1].plot(p, h_inh, "b-", linewidth=2, label="Hill (n=3)")
    axes[1].plot(p, s_inh, "r--", linewidth=2, label="PWL Step")
    axes[1].axvline(
        x=params.thetaB, color="gray", linestyle=":", label=f"θ = {params.thetaB} M"
    )
    axes[1].set_xlabel("Protein Concentration [M]")
    axes[1].set_ylabel("Inhibition Level")
    axes[1].set_title("Inhibition: Hill vs PWL Step")
    axes[1].legend()
    axes[1].set_xlim(0, 1.5)
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def plot_time_series_proteins(sol, save_path: str = None):
    """Plot protein concentrations vs time."""
    fig, ax = plt.subplots()

    ax.plot(sol.t, sol.y[2], color=COLOR_PA, label="Protein A (GUARDIAN)")
    ax.plot(sol.t, sol.y[3], color=COLOR_PB, label="Protein B (PROLIFERATOR)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Protein Concentration [M]")
    ax.set_title("PWL Model: Protein Time Series")
    ax.legend(loc="best")
    ax.set_xlim(sol.t[0], sol.t[-1])
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_time_series_mrna(sol, save_path: str = None):
    """Plot mRNA concentrations vs time."""
    fig, ax = plt.subplots()

    ax.plot(sol.t, sol.y[0], color=COLOR_RA, label="mRNA A")
    ax.plot(sol.t, sol.y[1], color=COLOR_RB, label="mRNA B")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("mRNA Concentration [M]")
    ax.set_title("PWL Model: mRNA Time Series")
    ax.legend(loc="best")
    ax.set_xlim(sol.t[0], sol.t[-1])
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_time_series_all(sol, save_path: str = None):
    """Plot all state variables vs time in subplots."""
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


def plot_phase_portrait_proteins(sol, params: PWLParameters, save_path: str = None):
    """Plot phase portrait in Protein A - Protein B plane."""
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

    # Draw threshold lines (important for PWL)
    ax.axvline(
        x=params.thetaA,
        color="blue",
        linestyle="--",
        alpha=0.5,
        label=f"$\\theta_A = {params.thetaA}$ M",
    )
    ax.axhline(
        y=params.thetaB,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label=f"$\\theta_B = {params.thetaB}$ M",
    )

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title("PWL Model: Phase Portrait (Protein Space)")
    ax.legend(loc="best")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_vector_field_proteins(
    params: PWLParameters,
    p_range: Tuple[float, float] = (0, 3),
    n_grid: int = 20,
    steepness: float = 100,
    save_path: str = None,
):
    """Plot vector field in protein phase space with PWL functions."""
    fig, ax = plt.subplots()

    # Create meshgrid
    pA = np.linspace(p_range[0] + 0.01, p_range[1], n_grid)
    pB = np.linspace(p_range[0] + 0.01, p_range[1], n_grid)
    PA, PB = np.meshgrid(pA, pB)

    # Quasi-steady-state mRNA approximation
    RA_ss = (params.mA / params.gammaA) * step_activation(PB, params.thetaB, steepness)
    RB_ss = (params.mB / params.gammaB) * step_inhibition(PA, params.thetaA, steepness)

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

    # Draw threshold lines (domain boundaries in PWL)
    ax.axvline(
        x=params.thetaA,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"$\\theta_A = {params.thetaA}$ M (switching surface)",
    )
    ax.axhline(
        y=params.thetaB,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"$\\theta_B = {params.thetaB}$ M (switching surface)",
    )

    # Label regulatory domains
    ax.text(
        0.05,
        0.05,
        "D1\n(A off, B on)",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.text(
        0.7,
        0.05,
        "D2\n(A off, B off)",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.text(
        0.05,
        0.85,
        "D3\n(A on, B on)",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.text(
        0.7,
        0.85,
        "D4\n(A on, B off)",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title("PWL Model: Vector Field with Switching Surfaces")
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
    params: PWLParameters,
    initial_conditions: List[List[float]] = None,
    t_span: Tuple[float, float] = (0, 50),
    save_path: str = None,
):
    """Plot multiple trajectories from different initial conditions."""
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
        sol = solve_pwl(params, y0=y0, t_span=t_span)
        ax.plot(
            sol.y[2],
            sol.y[3],
            color=colors[i],
            linewidth=1.5,
            label=f"IC: $p_A$={y0[2]:.1f}, $p_B$={y0[3]:.1f}",
        )
        ax.plot(sol.y[2][0], sol.y[3][0], "o", color=colors[i], markersize=8)
        ax.plot(sol.y[2][-1], sol.y[3][-1], "*", color=colors[i], markersize=12)

    # Draw threshold lines
    ax.axvline(x=params.thetaA, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=params.thetaB, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title("PWL Model: Multiple Trajectories")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def print_steady_state(sol):
    """Print the approximate steady-state values."""
    print("\n" + "=" * 50)
    print("PWL MODEL - STEADY-STATE VALUES (approximate)")
    print("=" * 50)
    print(f"mRNA A (rA):     {sol.y[0][-1]:.4f} M")
    print(f"mRNA B (rB):     {sol.y[1][-1]:.4f} M")
    print(f"Protein A (pA):  {sol.y[2][-1]:.4f} M")
    print(f"Protein B (pB):  {sol.y[3][-1]:.4f} M")
    print("=" * 50 + "\n")


def analyze_regulatory_domains(params: PWLParameters):
    """
    Analyze equilibria in each regulatory domain.

    PWL model divides phase space into 4 domains based on thresholds:
    - D1: pA < θA, pB < θB (Gene A off, Gene B on)
    - D2: pA > θA, pB < θB (Gene A off, Gene B off)
    - D3: pA < θA, pB > θB (Gene A on, Gene B on)
    - D4: pA > θA, pB > θB (Gene A on, Gene B off)
    """
    print("\n" + "=" * 60)
    print("PWL MODEL - REGULATORY DOMAIN ANALYSIS")
    print("=" * 60)

    # Calculate equilibria in each domain
    # At equilibrium: rA = (mA/γA)*s+(pB), pA = (kPA/δPA)*rA
    #                 rB = (mB/γB)*s-(pA), pB = (kPB/δPB)*rB

    k = params.kPA * params.mA / (params.deltaPA * params.gammaA)  # Combined constant

    # Domain 1: s+(pB)=0, s-(pA)=1 → pA*=0, pB*=k
    eq1_pA = 0
    eq1_pB = k
    print(f"\nD1 (pA<θA, pB<θB): Gene A OFF, Gene B ON")
    print(f"   Equilibrium: pA*={eq1_pA:.3f} M, pB*={eq1_pB:.3f} M")
    print(f"   Valid if pA*<{params.thetaA} and pB*<{params.thetaB}: ", end="")
    print("YES" if eq1_pA < params.thetaA and eq1_pB < params.thetaB else "NO")

    # Domain 2: s+(pB)=0, s-(pA)=0 → pA*=0, pB*=0
    eq2_pA = 0
    eq2_pB = 0
    print(f"\nD2 (pA>θA, pB<θB): Gene A OFF, Gene B OFF")
    print(f"   Equilibrium: pA*={eq2_pA:.3f} M, pB*={eq2_pB:.3f} M")
    print(f"   Valid if pA*>{params.thetaA} and pB*<{params.thetaB}: ", end="")
    print("YES" if eq2_pA > params.thetaA and eq2_pB < params.thetaB else "NO")

    # Domain 3: s+(pB)=1, s-(pA)=1 → pA*=k, pB*=k
    eq3_pA = k
    eq3_pB = k
    print(f"\nD3 (pA<θA, pB>θB): Gene A ON, Gene B ON")
    print(f"   Equilibrium: pA*={eq3_pA:.3f} M, pB*={eq3_pB:.3f} M")
    print(f"   Valid if pA*<{params.thetaA} and pB*>{params.thetaB}: ", end="")
    print("YES" if eq3_pA < params.thetaA and eq3_pB > params.thetaB else "NO")

    # Domain 4: s+(pB)=1, s-(pA)=0 → pA*=k, pB*=0
    eq4_pA = k
    eq4_pB = 0
    print(f"\nD4 (pA>θA, pB>θB): Gene A ON, Gene B OFF")
    print(f"   Equilibrium: pA*={eq4_pA:.3f} M, pB*={eq4_pB:.3f} M")
    print(f"   Valid if pA*>{params.thetaA} and pB*>{params.thetaB}: ", end="")
    print("YES" if eq4_pA > params.thetaA and eq4_pB > params.thetaB else "NO")

    print("=" * 60 + "\n")


# ===============================
# MAIN EXECUTION
# ===============================
def main():
    """Run the PWL Model simulation and generate plots."""

    print("\n" + "=" * 60)
    print("PIECEWISE-LINEAR (PWL) MODEL")
    print("Gene Regulatory Network: Activator-Inhibitor")
    print("=" * 60)

    # Initialize parameters
    params = PWLParameters()

    # Initial conditions [rA, rB, pA, pB] in M
    y0 = [0.8, 0.8, 0.8, 0.8]

    # Solve the system
    print("\nSolving PWL ODE system...")
    sol = solve_pwl(params, y0=y0, t_span=(0, 50), n_points=2000)

    # Print steady-state values
    print_steady_state(sol)

    # Analyze regulatory domains
    analyze_regulatory_domains(params)

    # Generate all plots
    print("Generating plots...")

    # 1. Step functions comparison
    plot_step_functions(params, save_path="output/pwl_step_functions.eps")

    # 2. Time series - Proteins only
    plot_time_series_proteins(sol, save_path="output/pwl_protein_timeseries.eps")

    # 3. Time series - mRNA only
    plot_time_series_mrna(sol, save_path="output/pwl_mrna_timeseries.eps")

    # 4. Time series - All variables
    plot_time_series_all(sol, save_path="output/pwl_all_timeseries.eps")

    # 5. Phase portrait
    plot_phase_portrait_proteins(sol, params, save_path="output/pwl_phase_portrait.eps")

    # 6. Vector field
    plot_vector_field_proteins(params, save_path="output/pwl_vector_field.eps")

    # 7. Multiple trajectories
    plot_multiple_trajectories(params, save_path="output/pwl_multiple_trajectories.eps")

    print("\nAll plots generated successfully!")
    print("Files saved in current directory with .eps format")

    # Show all plots
    plt.show()

    return sol, params


if __name__ == "__main__":
    sol, params = main()
