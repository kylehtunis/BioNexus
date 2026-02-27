"""
Bifurcation Analysis for Gene Regulatory Network Models
Investigates how steady-states change with varying parameters.

Key bifurcation parameters:
- Hill coefficient (n): Controls steepness of transcription response
- Threshold (θ): Expression threshold for protein binding
- Transcription rate (m): Maximum transcription rate
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ===============================
# PLOT CONFIGURATION
# ===============================
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
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


# ===============================
# HILL FUNCTIONS
# ===============================
def hill_activation(p, theta, n):
    """Hill activation function."""
    if n == 0:
        return 0.5
    return np.power(p, n) / (np.power(p, n) + np.power(theta, n))


def hill_inhibition(p, theta, n):
    """Hill inhibition function."""
    if n == 0:
        return 0.5
    return np.power(theta, n) / (np.power(p, n) + np.power(theta, n))


# ===============================
# STEADY-STATE EQUATIONS
# ===============================
def cnm_steady_state_equations(y, params):
    """
    Steady-state equations for CNM (setting all derivatives to zero).

    At steady state:
        rA = (mA/γA) * h+(pB, θB, nB)
        rB = (mB/γB) * h-(pA, θA, nA)
        pA = (kPA/δPA) * rA
        pB = (kPB/δPB) * rB

    Substituting:
        pA = (kPA*mA)/(δPA*γA) * h+(pB, θB, nB)
        pB = (kPB*mB)/(δPB*γB) * h-(pA, θA, nA)
    """
    pA, pB = y
    mA, mB, gammaA, gammaB, kPA, kPB, thetaA, thetaB, nA, nB, deltaPA, deltaPB = params

    # Constants
    KA = (kPA * mA) / (deltaPA * gammaA)
    KB = (kPB * mB) / (deltaPB * gammaB)

    # Steady-state conditions (residuals should be zero)
    eq1 = pA - KA * hill_activation(pB, thetaB, nB)
    eq2 = pB - KB * hill_inhibition(pA, thetaA, nA)

    return [eq1, eq2]


def find_steady_states(params, initial_guesses=None):
    """
    Find steady states by solving the algebraic equations.

    Args:
        params: Model parameters tuple
        initial_guesses: List of initial guesses for [pA, pB]

    Returns:
        List of unique steady states
    """
    if initial_guesses is None:
        initial_guesses = [
            [0.1, 0.1],
            [0.5, 0.5],
            [1.0, 1.0],
            [2.0, 0.1],
            [0.1, 2.0],
            [2.0, 2.0],
            [1.5, 0.5],
            [0.5, 1.5],
        ]

    steady_states = []
    for guess in initial_guesses:
        try:
            sol = fsolve(
                cnm_steady_state_equations, guess, args=(params,), full_output=True
            )
            x, info, ier, msg = sol
            if ier == 1:  # Solution found
                # Check if positive
                if x[0] >= 0 and x[1] >= 0:
                    # Check if unique (not already found)
                    is_unique = True
                    for ss in steady_states:
                        if np.allclose(x, ss, atol=1e-4):
                            is_unique = False
                            break
                    if is_unique:
                        steady_states.append(x)
        except:
            continue

    return steady_states


def compute_jacobian(pA, pB, params):
    """
    Compute Jacobian matrix at a steady state for stability analysis.

    For the reduced system (quasi-steady-state mRNA):
        dpA/dt = KA * h+(pB) - pA
        dpB/dt = KB * h-(pA) - pB

    Jacobian:
        J = [[-δPA, KA * dh+/dpB],
             [KB * dh-/dpA, -δPB]]
    """
    mA, mB, gammaA, gammaB, kPA, kPB, thetaA, thetaB, nA, nB, deltaPA, deltaPB = params

    KA = (kPA * mA) / (deltaPA * gammaA)
    KB = (kPB * mB) / (deltaPB * gammaB)

    # Derivatives of Hill functions
    # dh+/dp = n * theta^n * p^(n-1) / (p^n + theta^n)^2
    if nB > 0 and pB > 0:
        dh_act = nB * (thetaB**nB) * (pB ** (nB - 1)) / ((pB**nB + thetaB**nB) ** 2)
    else:
        dh_act = 0

    # dh-/dp = -n * theta^n * p^(n-1) / (p^n + theta^n)^2
    if nA > 0 and pA > 0:
        dh_inh = -nA * (thetaA**nA) * (pA ** (nA - 1)) / ((pA**nA + thetaA**nA) ** 2)
    else:
        dh_inh = 0

    J = np.array([[-deltaPA, KA * dh_act], [KB * dh_inh, -deltaPB]])

    return J


def analyze_stability(pA, pB, params):
    """
    Analyze stability of a steady state using eigenvalues.

    Returns:
        'stable': All eigenvalues have negative real parts
        'unstable': At least one eigenvalue has positive real part
        'saddle': Eigenvalues have opposite signs (real parts)
    """
    J = compute_jacobian(pA, pB, params)
    eigenvalues = np.linalg.eigvals(J)

    real_parts = np.real(eigenvalues)

    if all(r < 0 for r in real_parts):
        return "stable", eigenvalues
    elif all(r > 0 for r in real_parts):
        return "unstable", eigenvalues
    else:
        return "saddle", eigenvalues


# ===============================
# BIFURCATION DIAGRAMS
# ===============================
def bifurcation_hill_coefficient(n_range=(1, 10), n_points=50, save_path=None):
    """
    Bifurcation diagram varying Hill coefficient n.

    Shows how steady-state protein concentrations change with n.
    """
    # Base parameters
    mA, mB = 2.35, 2.35
    gammaA, gammaB = 1.0, 1.0
    kPA, kPB = 1.0, 1.0
    thetaA, thetaB = 0.21, 0.21
    deltaPA, deltaPB = 1.0, 1.0

    n_values = np.linspace(n_range[0], n_range[1], n_points)

    # Storage for results
    pA_stable = []
    pB_stable = []
    pA_unstable = []
    pB_unstable = []
    n_stable = []
    n_unstable = []

    for n in n_values:
        params = (
            mA,
            mB,
            gammaA,
            gammaB,
            kPA,
            kPB,
            thetaA,
            thetaB,
            n,
            n,
            deltaPA,
            deltaPB,
        )
        steady_states = find_steady_states(params)

        for ss in steady_states:
            pA, pB = ss
            stability, eigenvalues = analyze_stability(pA, pB, params)

            if stability == "stable":
                pA_stable.append(pA)
                pB_stable.append(pB)
                n_stable.append(n)
            else:
                pA_unstable.append(pA)
                pB_unstable.append(pB)
                n_unstable.append(n)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Protein A
    if n_stable:
        axes[0].scatter(n_stable, pA_stable, c="blue", s=20, label="Stable")
    if n_unstable:
        axes[0].scatter(
            n_unstable, pA_unstable, c="red", s=20, marker="x", label="Unstable"
        )
    axes[0].axhline(
        y=thetaA, color="gray", linestyle="--", alpha=0.5, label=f"θ = {thetaA}"
    )
    axes[0].set_xlabel("Hill Coefficient (n)")
    axes[0].set_ylabel("Protein A Steady-State [M]")
    axes[0].set_title("Bifurcation Diagram: Protein A vs Hill Coefficient")
    axes[0].legend()
    axes[0].set_xlim(n_range)

    # Protein B
    if n_stable:
        axes[1].scatter(n_stable, pB_stable, c="blue", s=20, label="Stable")
    if n_unstable:
        axes[1].scatter(
            n_unstable, pB_unstable, c="red", s=20, marker="x", label="Unstable"
        )
    axes[1].axhline(
        y=thetaB, color="gray", linestyle="--", alpha=0.5, label=f"θ = {thetaB}"
    )
    axes[1].set_xlabel("Hill Coefficient (n)")
    axes[1].set_ylabel("Protein B Steady-State [M]")
    axes[1].set_title("Bifurcation Diagram: Protein B vs Hill Coefficient")
    axes[1].legend()
    axes[1].set_xlim(n_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def bifurcation_threshold(theta_range=(0.05, 1.0), theta_points=50, save_path=None):
    """
    Bifurcation diagram varying threshold θ (same for both genes).
    """
    # Base parameters
    mA, mB = 2.35, 2.35
    gammaA, gammaB = 1.0, 1.0
    kPA, kPB = 1.0, 1.0
    nA, nB = 3, 3
    deltaPA, deltaPB = 1.0, 1.0

    theta_values = np.linspace(theta_range[0], theta_range[1], theta_points)

    # Storage
    pA_stable, pB_stable = [], []
    pA_unstable, pB_unstable = [], []
    theta_stable, theta_unstable = [], []

    for theta in theta_values:
        params = (
            mA,
            mB,
            gammaA,
            gammaB,
            kPA,
            kPB,
            theta,
            theta,
            nA,
            nB,
            deltaPA,
            deltaPB,
        )
        steady_states = find_steady_states(params)

        for ss in steady_states:
            pA, pB = ss
            stability, _ = analyze_stability(pA, pB, params)

            if stability == "stable":
                pA_stable.append(pA)
                pB_stable.append(pB)
                theta_stable.append(theta)
            else:
                pA_unstable.append(pA)
                pB_unstable.append(pB)
                theta_unstable.append(theta)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Protein A
    if theta_stable:
        axes[0].scatter(theta_stable, pA_stable, c="blue", s=20, label="Stable")
    if theta_unstable:
        axes[0].scatter(
            theta_unstable, pA_unstable, c="red", s=20, marker="x", label="Unstable"
        )
    # Reference line (pA = theta)
    axes[0].plot(theta_values, theta_values, "g--", alpha=0.5, label="$p_A = \\theta$")
    axes[0].set_xlabel("Threshold θ [M]")
    axes[0].set_ylabel("Protein A Steady-State [M]")
    axes[0].set_title("Bifurcation Diagram: Protein A vs Threshold")
    axes[0].legend()

    # Protein B
    if theta_stable:
        axes[1].scatter(theta_stable, pB_stable, c="blue", s=20, label="Stable")
    if theta_unstable:
        axes[1].scatter(
            theta_unstable, pB_unstable, c="red", s=20, marker="x", label="Unstable"
        )
    axes[1].plot(theta_values, theta_values, "g--", alpha=0.5, label="$p_B = \\theta$")
    axes[1].set_xlabel("Threshold θ [M]")
    axes[1].set_ylabel("Protein B Steady-State [M]")
    axes[1].set_title("Bifurcation Diagram: Protein B vs Threshold")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def bifurcation_transcription_rate(m_range=(0.5, 5.0), m_points=50, save_path=None):
    """
    Bifurcation diagram varying transcription rate m (same for both genes).
    """
    # Base parameters
    gammaA, gammaB = 1.0, 1.0
    kPA, kPB = 1.0, 1.0
    thetaA, thetaB = 0.21, 0.21
    nA, nB = 3, 3
    deltaPA, deltaPB = 1.0, 1.0

    m_values = np.linspace(m_range[0], m_range[1], m_points)

    # Storage
    pA_stable, pB_stable = [], []
    pA_unstable, pB_unstable = [], []
    m_stable, m_unstable = [], []

    for m in m_values:
        params = (
            m,
            m,
            gammaA,
            gammaB,
            kPA,
            kPB,
            thetaA,
            thetaB,
            nA,
            nB,
            deltaPA,
            deltaPB,
        )
        steady_states = find_steady_states(params)

        for ss in steady_states:
            pA, pB = ss
            stability, _ = analyze_stability(pA, pB, params)

            if stability == "stable":
                pA_stable.append(pA)
                pB_stable.append(pB)
                m_stable.append(m)
            else:
                pA_unstable.append(pA)
                pB_unstable.append(pB)
                m_unstable.append(m)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if m_stable:
        axes[0].scatter(m_stable, pA_stable, c="blue", s=20, label="Stable")
        axes[1].scatter(m_stable, pB_stable, c="blue", s=20, label="Stable")
    if m_unstable:
        axes[0].scatter(
            m_unstable, pA_unstable, c="red", s=20, marker="x", label="Unstable"
        )
        axes[1].scatter(
            m_unstable, pB_unstable, c="red", s=20, marker="x", label="Unstable"
        )

    axes[0].axhline(
        y=thetaA, color="gray", linestyle="--", alpha=0.5, label=f"θ = {thetaA}"
    )
    axes[0].set_xlabel("Transcription Rate m [s⁻¹]")
    axes[0].set_ylabel("Protein A Steady-State [M]")
    axes[0].set_title("Bifurcation Diagram: Protein A vs Transcription Rate")
    axes[0].legend()

    axes[1].axhline(
        y=thetaB, color="gray", linestyle="--", alpha=0.5, label=f"θ = {thetaB}"
    )
    axes[1].set_xlabel("Transcription Rate m [s⁻¹]")
    axes[1].set_ylabel("Protein B Steady-State [M]")
    axes[1].set_title("Bifurcation Diagram: Protein B vs Transcription Rate")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


def bifurcation_2d_phase_diagram(
    n_range=(1, 8), theta_range=(0.1, 0.5), resolution=30, save_path=None
):
    """
    2D bifurcation diagram showing regions of different behavior
    in (n, θ) parameter space.
    """
    # Base parameters
    mA, mB = 2.35, 2.35
    gammaA, gammaB = 1.0, 1.0
    kPA, kPB = 1.0, 1.0
    deltaPA, deltaPB = 1.0, 1.0

    n_values = np.linspace(n_range[0], n_range[1], resolution)
    theta_values = np.linspace(theta_range[0], theta_range[1], resolution)

    # Count number of stable steady states for each parameter combination
    num_stable = np.zeros((resolution, resolution))

    for i, n in enumerate(n_values):
        for j, theta in enumerate(theta_values):
            params = (
                mA,
                mB,
                gammaA,
                gammaB,
                kPA,
                kPB,
                theta,
                theta,
                n,
                n,
                deltaPA,
                deltaPB,
            )
            steady_states = find_steady_states(params)

            count = 0
            for ss in steady_states:
                pA, pB = ss
                stability, _ = analyze_stability(pA, pB, params)
                if stability == "stable":
                    count += 1
            num_stable[j, i] = count

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        num_stable,
        extent=[n_range[0], n_range[1], theta_range[0], theta_range[1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )

    ax.set_xlabel("Hill Coefficient (n)")
    ax.set_ylabel("Threshold θ [M]")
    ax.set_title("2D Bifurcation Diagram: Number of Stable Steady States")

    # Mark current parameter values
    ax.axvline(x=3, color="red", linestyle="--", linewidth=2, label="Assignment: n=3")
    ax.axhline(
        y=0.21, color="red", linestyle=":", linewidth=2, label="Assignment: θ=0.21"
    )
    ax.plot(3, 0.21, "r*", markersize=20, label="Current Parameters")

    ax.legend(loc="upper right")

    cbar = plt.colorbar(im, label="Number of Stable Equilibria")
    cbar.set_ticks([0, 1, 2, 3])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_nullclines_with_equilibria(n=3, theta=0.21, save_path=None):
    """
    Plot nullclines and mark equilibrium points with stability.
    """
    # Parameters
    mA, mB = 2.35, 2.35
    gammaA, gammaB = 1.0, 1.0
    kPA, kPB = 1.0, 1.0
    deltaPA, deltaPB = 1.0, 1.0

    params = (mA, mB, gammaA, gammaB, kPA, kPB, theta, theta, n, n, deltaPA, deltaPB)

    KA = (kPA * mA) / (deltaPA * gammaA)
    KB = (kPB * mB) / (deltaPB * gammaB)

    fig, ax = plt.subplots(figsize=(10, 8))

    # pA nullcline: pA = KA * h+(pB)
    pB_range = np.linspace(0.001, 3, 500)
    pA_nullcline = KA * hill_activation(pB_range, theta, n)
    ax.plot(pA_nullcline, pB_range, "b-", linewidth=2, label="$dp_A/dt = 0$")

    # pB nullcline: pB = KB * h-(pA)
    pA_range = np.linspace(0.001, 3, 500)
    pB_nullcline = KB * hill_inhibition(pA_range, theta, n)
    ax.plot(pA_range, pB_nullcline, "r-", linewidth=2, label="$dp_B/dt = 0$")

    # Find and plot equilibria
    steady_states = find_steady_states(params)

    for ss in steady_states:
        pA, pB = ss
        stability, eigenvalues = analyze_stability(pA, pB, params)

        if stability == "stable":
            ax.plot(
                pA,
                pB,
                "go",
                markersize=15,
                markeredgecolor="black",
                markeredgewidth=2,
                label=f"Stable: ({pA:.2f}, {pB:.2f})",
                zorder=10,
            )
        else:
            ax.plot(
                pA,
                pB,
                "ro",
                markersize=15,
                markeredgecolor="black",
                markeredgewidth=2,
                label=f"{stability}: ({pA:.2f}, {pB:.2f})",
                zorder=10,
            )

        # Print eigenvalue info
        print(f"Equilibrium at (pA={pA:.4f}, pB={pB:.4f}): {stability}")
        print(f"  Eigenvalues: {eigenvalues}")

    # Threshold lines
    ax.axvline(x=theta, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=theta, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title(f"Nullclines and Equilibria (n={n}, θ={theta})")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


# ===============================
# MAIN EXECUTION
# ===============================
def main():
    """Generate all bifurcation diagrams."""

    print("\n" + "=" * 60)
    print("BIFURCATION ANALYSIS")
    print("Gene Regulatory Network: Activator-Inhibitor")
    print("=" * 60)

    # 1. Nullclines with equilibria (current parameters)
    print("\n1. Plotting nullclines and equilibria...")
    plot_nullclines_with_equilibria(
        n=3, theta=0.21, save_path="output/bifurcation_nullclines.eps"
    )

    # 2. Bifurcation vs Hill coefficient
    print("\n2. Bifurcation diagram: Hill coefficient...")
    bifurcation_hill_coefficient(
        n_range=(1, 10), save_path="output/bifurcation_hill_coeff.eps"
    )

    # 3. Bifurcation vs threshold
    print("\n3. Bifurcation diagram: Threshold...")
    bifurcation_threshold(
        theta_range=(0.05, 1.5), save_path="output/bifurcation_threshold.eps"
    )

    # 4. Bifurcation vs transcription rate
    print("\n4. Bifurcation diagram: Transcription rate...")
    bifurcation_transcription_rate(
        m_range=(0.5, 5.0), save_path="output/bifurcation_transcription.eps"
    )

    # 5. 2D bifurcation phase diagram
    print("\n5. 2D bifurcation phase diagram...")
    bifurcation_2d_phase_diagram(save_path="output/bifurcation_2d_phase.eps")

    print("\n" + "=" * 60)
    print("All bifurcation diagrams generated!")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
