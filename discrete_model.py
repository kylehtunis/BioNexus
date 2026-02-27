"""
Discrete-Time Model for Gene Regulatory Network
Based on the activator-inhibitor network from Polynikis et al. (2009)

Paper Eq. 23: Exact exponential integration of the SPWLM:
    pa(n+1) = exp(-da)·pa(n) + (ka'/da)·(1 - exp(-da))·s+(pb(n), θb)
    pb(n+1) = exp(-db)·pb(n) + (kb'/db)·(1 - exp(-db))·s-(pa(n), θa)

where ka' = (ma/γa)·ka, kb' = (mb/γb)·kb

This is derived from the Simplified PWL Model (SPWLM) using quasi-steady-state
mRNA assumption + PWL step functions + exact integration of the linear part.

Gene A (GUARDIAN): Tumor-suppressor - activated by Protein B
Gene B (PROLIFERATOR): Oncogenic - inhibited by Protein A
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

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


# ===============================
# MODEL PARAMETERS
# ===============================
class DiscreteParameters:
    """Parameters for the Discrete-Time Model (Paper Eq. 23)"""

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

    @property
    def kA_prime(self):
        """Combined constant ka' = (ma/γa)·ka (Paper Eq. 13)"""
        return (self.mA / self.gammaA) * self.kPA

    @property
    def kB_prime(self):
        """Combined constant kb' = (mb/γb)·kb (Paper Eq. 13)"""
        return (self.mB / self.gammaB) * self.kPB

    @property
    def alpha(self):
        """Decay factor a = exp(-da) (Paper Eq. 25)"""
        return np.exp(-self.deltaPA)

    @property
    def T(self):
        """Timestep T = 1/(10·max(1/da, 1/db)) (Paper Eq. 21)"""
        return 1.0 / (10 * max(1.0 / self.deltaPA, 1.0 / self.deltaPB))


# ===============================
# PWL STEP FUNCTIONS
# ===============================
def step_activation(p, theta):
    """Step function for activation: s+(p,θ) = 0 if p<θ, 1 if p>θ"""
    return np.where(np.asarray(p) > theta, 1.0, 0.0)


def step_inhibition(p, theta):
    """Step function for inhibition: s-(p,θ) = 1 if p<θ, 0 if p>θ"""
    return np.where(np.asarray(p) < theta, 1.0, 0.0)


# ===============================
# DISCRETE-TIME MODEL (Paper Eq. 23)
# ===============================
def discrete_step_paper(
    pA: float, pB: float, params: DiscreteParameters
) -> Tuple[float, float]:
    """
    One discrete time step per Paper Eq. 23 (exact exponential integration).

    pa(n+1) = exp(-da·T)·pa(n) + (ka'/da)·(1 - exp(-da·T))·s+(pb(n), θb)
    pb(n+1) = exp(-db·T)·pb(n) + (kb'/db)·(1 - exp(-db·T))·s-(pa(n), θa)

    Uses rescaled time T=1 (Paper Eq. 21-23), giving:
    pa(n+1) = exp(-da)·pa(n) + (ka'/da)·(1 - exp(-da))·s+(pb(n), θb)
    pb(n+1) = exp(-db)·pb(n) + (kb'/db)·(1 - exp(-db))·s-(pa(n), θa)

    Args:
        pA: Current Protein A concentration [M]
        pB: Current Protein B concentration [M]
        params: Model parameters

    Returns:
        (pA_next, pB_next) concentrations at next timestep
    """
    da = params.deltaPA
    db = params.deltaPB

    exp_da = np.exp(-da)
    exp_db = np.exp(-db)

    # PWL step functions
    s_act = float(step_activation(pB, params.thetaB))
    s_inh = float(step_inhibition(pA, params.thetaA))

    # Paper Eq. 23
    pA_next = exp_da * pA + (params.kA_prime / da) * (1 - exp_da) * s_act
    pB_next = exp_db * pB + (params.kB_prime / db) * (1 - exp_db) * s_inh

    return pA_next, pB_next


def solve_discrete(
    params: DiscreteParameters,
    y0: List[float] = [0.8, 0.8, 0.8, 0.8],
    t_span: Tuple[float, float] = (0, 50),
    dt: float = 0.1,
    use_pwl: bool = True,
):
    """
    Solve the discrete-time model (Paper Eq. 23).

    Note: This is a 2-variable model (proteins only) derived from the SPWLM.
    The mRNA state is not tracked (quasi-steady-state assumption).

    Args:
        params: Model parameters
        y0: Initial conditions [rA0, rB0, pA0, pB0] (only pA0, pB0 used)
        t_span: Time interval [s]
        dt: Not used (timestep is T=1 in rescaled time); kept for API compatibility
        use_pwl: Not used; always uses PWL (kept for API compatibility)

    Returns:
        t: Discrete time step indices
        y: State array (4 x n_steps), rows 0,1 are mRNA (reconstructed), 2,3 are proteins
    """
    # Number of discrete steps (each step = 1 in rescaled time)
    t_start, t_end = t_span
    n_steps = int(t_end - t_start) + 1

    t = np.arange(n_steps)
    pA = np.zeros(n_steps)
    pB = np.zeros(n_steps)

    # Initial conditions (use protein values from y0)
    pA[0] = y0[2]
    pB[0] = y0[3]

    for k in range(n_steps - 1):
        pA[k + 1], pB[k + 1] = discrete_step_paper(pA[k], pB[k], params)

    # Reconstruct mRNA from quasi-steady-state assumption (Paper Eq. 11)
    rA = (params.mA / params.gammaA) * step_activation(pB, params.thetaB)
    rB = (params.mB / params.gammaB) * step_inhibition(pA, params.thetaA)

    # Return in same format as continuous models for compatibility
    y = np.array([rA, rB, pA, pB])

    return t, y


# ===============================
# ANALYSIS FUNCTIONS
# ===============================
def analyze_discrete_parameters(params: DiscreteParameters):
    """
    Print the derived parameters for the discrete model.
    Paper Eq. 13, 21, 24, 25.
    """
    da = params.deltaPA
    db = params.deltaPB
    alpha = params.alpha

    print("\n" + "=" * 50)
    print("DISCRETE MODEL PARAMETERS (Paper Eq. 13, 25)")
    print("=" * 50)
    print(f"ka' = (mA/γA)·kPA = {params.kA_prime:.4f}")
    print(f"kb' = (mB/γB)·kPB = {params.kB_prime:.4f}")
    print(f"α = exp(-da) = {alpha:.4f}")
    print(f"T = 1/(10·max(1/da, 1/db)) = {params.T:.4f} s")
    print()
    print("Since da = ka' = db = kb' = 1.0, Paper Eq. 24 applies:")
    print(f"  pa(n+1) = α·pa(n) + (1-α)·s+(pb(n), θb)")
    print(f"  pb(n+1) = α·pb(n) + (1-α)·s-(pa(n), θa)")
    print(f"  with α = {alpha:.4f}")

    # But ka' = mA/gammaA * kPA = 2.35, not 1.0!
    # So Eq. 24 does NOT apply since da ≠ ka'
    if abs(da - params.kA_prime) > 0.01:
        print()
        print("  ⚠️  However, da ≠ ka' (1.0 ≠ 2.35), so Eq. 24 does NOT simplify.")
        print("  Using the general form Eq. 23 instead:")
        coeff_A = (params.kA_prime / da) * (1 - np.exp(-da))
        coeff_B = (params.kB_prime / db) * (1 - np.exp(-db))
        print(f"    pa(n+1) = {alpha:.4f}·pa(n) + {coeff_A:.4f}·s+(pb(n), θb)")
        print(f"    pb(n+1) = {alpha:.4f}·pb(n) + {coeff_B:.4f}·s-(pa(n), θa)")
    print("=" * 50 + "\n")


def compare_decay_factors(params: DiscreteParameters):
    """Compare solutions for different decay factors (like varying da)."""
    da_values = [0.5, 1.0, 2.0, 5.0]
    solutions = {}

    for da in da_values:
        p = DiscreteParameters()
        p.deltaPA = da
        p.deltaPB = da
        t, y = solve_discrete(p, y0=[0.8, 0.8, 0.8, 0.8], t_span=(0, 50))
        solutions[da] = (t, y)

    return solutions


# ===============================
# PLOTTING FUNCTIONS
# ===============================
def plot_time_series_proteins(t, y, save_path: str = None):
    """Plot protein concentrations vs discrete time step."""
    fig, ax = plt.subplots()

    ax.step(t, y[2], where="post", color=COLOR_PA, label="Protein A (GUARDIAN)")
    ax.step(t, y[3], where="post", color=COLOR_PB, label="Protein B (PROLIFERATOR)")
    ax.plot(t, y[2], "o", color=COLOR_PA, markersize=2, alpha=0.3)
    ax.plot(t, y[3], "o", color=COLOR_PB, markersize=2, alpha=0.3)

    ax.set_xlabel("Discrete Time Step (n)")
    ax.set_ylabel("Protein Concentration [M]")
    ax.set_title("Discrete Model (Paper Eq. 23): Protein Time Series")
    ax.legend(loc="best")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_phase_portrait_proteins(
    t, y, params: DiscreteParameters, save_path: str = None
):
    """Plot phase portrait in Protein A - Protein B plane."""
    fig, ax = plt.subplots()

    # Plot trajectory with discrete points
    ax.plot(y[2], y[3], "-", color="#9467bd", linewidth=1, alpha=0.5)
    ax.plot(
        y[2], y[3], "o", color="#9467bd", markersize=3, alpha=0.5, label="Trajectory"
    )

    # Mark start and end points
    ax.plot(y[2][0], y[3][0], "go", markersize=10, label="Start", zorder=5)
    ax.plot(y[2][-1], y[3][-1], "r*", markersize=15, label="End", zorder=5)

    # Switching surfaces
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
    ax.set_title("Discrete Model (Paper Eq. 23): Phase Portrait")
    ax.legend(loc="best")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_multiple_trajectories(
    params: DiscreteParameters,
    initial_conditions: List[List[float]] = None,
    t_span: Tuple[float, float] = (0, 50),
    save_path: str = None,
):
    """Plot multiple trajectories from different initial conditions."""
    if initial_conditions is None:
        initial_conditions = [
            [0, 0, 0.1, 0.1],
            [0, 0, 0.8, 0.8],
            [0, 0, 1.5, 0.5],
            [0, 0, 0.5, 1.5],
            [0, 0, 2.0, 2.0],
            [0, 0, 0.2, 1.0],
        ]

    fig, ax = plt.subplots()

    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions)))

    for i, y0 in enumerate(initial_conditions):
        t, y = solve_discrete(params, y0=y0, t_span=t_span)
        ax.plot(
            y[2],
            y[3],
            "-o",
            color=colors[i],
            linewidth=1,
            markersize=2,
            label=f"IC: $p_A$={y0[2]:.1f}, $p_B$={y0[3]:.1f}",
        )
        ax.plot(y[2][0], y[3][0], "o", color=colors[i], markersize=8)
        ax.plot(y[2][-1], y[3][-1], "*", color=colors[i], markersize=12)

    # Switching surfaces
    ax.axvline(x=params.thetaA, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=params.thetaB, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title("Discrete Model: Multiple Trajectories")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_decay_factor_comparison(solutions: dict, save_path: str = None):
    """Compare solutions for different protein degradation rates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(solutions)))

    for (da, (t, y)), color in zip(solutions.items(), colors):
        alpha = np.exp(-da)
        axes[0].step(
            t, y[2], where="post", color=color, label=f"$d_a$={da}, α={alpha:.3f}"
        )
        axes[1].step(
            t, y[3], where="post", color=color, label=f"$d_b$={da}, α={alpha:.3f}"
        )

    axes[0].set_xlabel("Discrete Time Step (n)")
    axes[0].set_ylabel("Protein A Concentration [M]")
    axes[0].set_title("Protein A: Decay Factor Comparison")
    axes[0].legend()

    axes[1].set_xlabel("Discrete Time Step (n)")
    axes[1].set_ylabel("Protein B Concentration [M]")
    axes[1].set_title("Protein B: Decay Factor Comparison")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def print_steady_state(t, y):
    """Print the approximate steady-state values."""
    print("\n" + "=" * 50)
    print("DISCRETE MODEL (Paper Eq. 23) - STEADY-STATE VALUES")
    print("=" * 50)
    print(f"mRNA A (rA):     {y[0][-1]:.4f} M  (quasi-steady-state)")
    print(f"mRNA B (rB):     {y[1][-1]:.4f} M  (quasi-steady-state)")
    print(f"Protein A (pA):  {y[2][-1]:.4f} M")
    print(f"Protein B (pB):  {y[3][-1]:.4f} M")
    print("=" * 50 + "\n")


# ===============================
# MAIN EXECUTION
# ===============================
def main():
    """Run the Discrete-Time Model simulation and generate plots."""

    print("\n" + "=" * 60)
    print("DISCRETE-TIME MODEL (Paper Eq. 23)")
    print("Gene Regulatory Network: Activator-Inhibitor")
    print("Exact Exponential Integration of SPWLM")
    print("=" * 60)

    # Initialize parameters
    params = DiscreteParameters()

    # Print derived parameters
    analyze_discrete_parameters(params)

    # Initial conditions [rA, rB, pA, pB] in M
    y0 = [0.8, 0.8, 0.8, 0.8]

    # Solve the system
    print("Solving discrete model (Paper Eq. 23)...")
    t, y = solve_discrete(params, y0=y0, t_span=(0, 50))

    # Print steady-state values
    print_steady_state(t, y)

    # Generate plots
    print("Generating plots...")

    # 1. Time series - Proteins
    plot_time_series_proteins(t, y, save_path="output/discrete_protein_timeseries.eps")

    # 2. Phase portrait
    plot_phase_portrait_proteins(
        t, y, params, save_path="output/discrete_phase_portrait.eps"
    )

    # 3. Multiple trajectories
    plot_multiple_trajectories(
        params, save_path="output/discrete_multiple_trajectories.eps"
    )

    # 4. Decay factor comparison
    print("\nComparing different decay factors...")
    solutions = compare_decay_factors(params)
    plot_decay_factor_comparison(
        solutions, save_path="output/discrete_decay_comparison.eps"
    )

    print("\nAll plots generated successfully!")
    print("Files saved in current directory with .eps format")

    # Show all plots
    plt.show()

    return t, y, params


if __name__ == "__main__":
    t, y, params = main()
