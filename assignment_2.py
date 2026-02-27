"""
Assignment 2 — Gene Regulatory Network Modeling & Comparative Diagnosis
=========================================================================
Viterbi HMM classification, ODE/SDE gene regulation models (CNM, PWL,
Discrete, SDEVelo), bifurcation analysis, and Lotka-Volterra bonus.
"""

import os
import warnings
from dataclasses import dataclass, field

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ══════════════════════════════════════════════════════════════════════
# HMM PARAMETERS (Assignment Tables 1–3) & VITERBI ALGORITHM
# ══════════════════════════════════════════════════════════════════════

# Table 1: Transition probabilities [Exon→Exon, Exon→Intron; Intron→Exon, Intron→Intron]
HMM_TRANSITIONS = np.array(
    [
        [0.9, 0.1],
        [0.2, 0.8],
    ]
)

# Table 2: Emission probabilities [state × nucleotide(A,U,G,C)]
HMM_EMISSIONS = np.array(
    [
        [0.25, 0.25, 0.25, 0.25],  # Exon: uniform
        [0.40, 0.40, 0.05, 0.15],  # Intron: A/U-rich
    ]
)

# Table 3: Initial probabilities
HMM_INITIAL = np.array([0.5, 0.5])

# Patient sequences (nucleotide-encoded)
SEQ_ALPHA = np.array([0, 2, 3, 2, 3])  # AGCGC
SEQ_BETA = np.array([0, 1, 1, 0, 1])  # AUUAU


def viterbi(
    sequence: np.ndarray,
    initial: np.ndarray,
    emissions: np.ndarray,
    transitions: np.ndarray,
    state: int = -1,
) -> tuple[float, str]:
    """Recursive Viterbi algorithm for HMM decoding."""
    if state < 0:
        v0, bt0 = viterbi(sequence, initial, emissions, transitions, 0)
        v1, bt1 = viterbi(sequence, initial, emissions, transitions, 1)
        return (v0, bt0 + "E") if v0 > v1 else (v1, bt1 + "I")

    emitted = sequence[-1]
    if sequence.size == 1:
        return initial[state] * emissions[state, emitted], ""

    v0, bt0 = viterbi(sequence[:-1], initial, emissions, transitions, 0)
    v0 *= transitions[state, 0] * emissions[state, emitted]
    v1, bt1 = viterbi(sequence[:-1], initial, emissions, transitions, 1)
    v1 *= transitions[state, 1] * emissions[state, emitted]
    return (v0, bt0 + "E") if v0 > v1 else (v1, bt1 + "I")


def classify_patients() -> tuple[str, str]:
    """Run Viterbi on both patients and return (mechanism_alpha, mechanism_beta)."""
    prob_a, path_a = viterbi(SEQ_ALPHA, HMM_INITIAL, HMM_EMISSIONS, HMM_TRANSITIONS)
    prob_b, path_b = viterbi(SEQ_BETA, HMM_INITIAL, HMM_EMISSIONS, HMM_TRANSITIONS)

    exon_count_a = path_a.count("E")
    exon_count_b = path_b.count("E")
    mech_a = "I" if exon_count_a > len(path_a) // 2 else "II"
    mech_b = "I" if exon_count_b > len(path_b) // 2 else "II"


    return mech_a, mech_b


# ══════════════════════════════════════════════════════════════════════
# PLOT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
matplotlib.use("Agg")  # Non-interactive backend for EPS generation
plt.rcParams["figure.max_open_warning"] = 0  # Suppress figure count warning

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

# Consistent colour palette across all models
COLOR_PA = "#1f77b4"  # Blue   — Protein A (GUARDIAN)
COLOR_PB = "#ff7f0e"  # Orange — Protein B (PROLIFERATOR)
COLOR_RA = "#2ca02c"  # Green  — mRNA A
COLOR_RB = "#d62728"  # Red    — mRNA B
COLOR_CNM = "#1f77b4"  # Blue   — CNM model traces
COLOR_PWL = "#ff7f0e"  # Orange — PWL model traces
COLOR_DIS = "#2ca02c"  # Green  — Discrete model traces

OUTPUT_DIR = "output"


def _save(fig, path: str | None):
    """Save figure to EPS if path is given."""
    if path:
        full = os.path.join(OUTPUT_DIR, path)
        fig.savefig(full, format="eps", bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════
# PARAMETERS  (Assignment Table 5 — shared by all three models)
# ══════════════════════════════════════════════════════════════════════
@dataclass
class Parameters:
    """Kinetic parameters for the activator-inhibitor gene regulatory network."""

    # Max transcription rates [s⁻¹]
    mA: float = 2.35
    mB: float = 2.35
    # mRNA degradation rates [s⁻¹]
    gammaA: float = 1.0
    gammaB: float = 1.0
    # Translation rates [s⁻¹]
    kPA: float = 1.0
    kPB: float = 1.0
    # Expression thresholds [M]  — concentration at half-max regulation
    thetaA: float = 0.21
    thetaB: float = 0.21
    # Hill coefficients (cooperativity index, dimensionless)
    nA: int = 3
    nB: int = 3
    # Protein degradation rates [s⁻¹]
    deltaPA: float = 1.0
    deltaPB: float = 1.0
    # Initial conditions [M]
    y0: list[float] = field(default_factory=lambda: [0.8, 0.8, 0.8, 0.8])

    # ── Derived constants for the Simplified/Discrete models ──────────

    @property
    def kA_prime(self) -> float:
        """Combined production constant ka' = (mA / γA) · kPA."""
        return (self.mA / self.gammaA) * self.kPA

    @property
    def kB_prime(self) -> float:
        """Combined production constant kb' = (mB / γB) · kPB = 2.35."""
        return (self.mB / self.gammaB) * self.kPB

    @property
    def alpha_A(self) -> float:
        """Discrete decay factor αA = exp(−δPA) ≈ 0.368 (Paper Eq. 25)."""
        return np.exp(-self.deltaPA)

    @property
    def alpha_B(self) -> float:
        """Discrete decay factor αB = exp(−δPB) ≈ 0.368."""
        return np.exp(-self.deltaPB)

    def as_tuple(self, n_override=None, theta_override=None, m_override=None):
        """Pack parameters into a tuple for bifurcation solvers."""
        n = n_override if n_override is not None else self.nA
        theta = theta_override if theta_override is not None else self.thetaA
        m = m_override if m_override is not None else self.mA
        return (
            m,
            m,
            self.gammaA,
            self.gammaB,
            self.kPA,
            self.kPB,
            theta,
            theta,
            n,
            n,
            self.deltaPA,
            self.deltaPB,
        )


# ══════════════════════════════════════════════════════════════════════
# REGULATORY (TRANSCRIPTION) FUNCTIONS
# ══════════════════════════════════════════════════════════════════════


def hill_activation(p: np.ndarray, theta: float, n: int) -> np.ndarray:
    """Hill activation function h⁺(p, θ, n) = pⁿ / (pⁿ + θⁿ)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.power(p, n) / (np.power(p, n) + np.power(theta, n))


def hill_inhibition(p: np.ndarray, theta: float, n: int) -> np.ndarray:
    """Hill inhibition function h⁻(p, θ, n) = θⁿ / (pⁿ + θⁿ)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.power(theta, n) / (np.power(p, n) + np.power(theta, n))


def step_activation_smooth(
    p: np.ndarray, theta: float, steepness: float = 100
) -> np.ndarray:
    """Smooth approximation of the PWL step function for activation."""
    return 1.0 / (1.0 + np.exp(-steepness * (p - theta)))


def step_inhibition_smooth(
    p: np.ndarray, theta: float, steepness: float = 100
) -> np.ndarray:
    """Smooth step inhibition: s⁻ = 1 − s⁺."""
    return 1.0 - step_activation_smooth(p, theta, steepness)


def step_activation_hard(p, theta) -> np.ndarray:
    """Hard (discontinuous) step function for the discrete model."""
    return np.where(np.asarray(p) > theta, 1.0, 0.0)


def step_inhibition_hard(p, theta) -> np.ndarray:
    """Hard step inhibition: s⁻ = 1 − s⁺."""
    return np.where(np.asarray(p) < theta, 1.0, 0.0)


# ══════════════════════════════════════════════════════════════════════
# MODEL 1: COMPLETE NONLINEAR MODEL  (CNM — Paper Eq. 9–10)
# ══════════════════════════════════════════════════════════════════════


def cnm_ode(t: float, y: np.ndarray, p: Parameters) -> list[float]:
    """CNM: four coupled ODEs for the activator-inhibitor network."""
    rA, rB, pA, pB = y
    drA = p.mA * hill_activation(pB, p.thetaB, p.nB) - p.gammaA * rA
    drB = p.mB * hill_inhibition(pA, p.thetaA, p.nA) - p.gammaB * rB
    dpA = p.kPA * rA - p.deltaPA * pA
    dpB = p.kPB * rB - p.deltaPB * pB
    return [drA, drB, dpA, dpB]


def solve_cnm(p: Parameters, t_span=(0, 200), n_points=5000):
    """Integrate the CNM system using Runge-Kutta 4(5)."""
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    return solve_ivp(
        lambda t, y: cnm_ode(t, y, p),
        t_span,
        p.y0,
        t_eval=t_eval,
        method="RK45",
        dense_output=True,
    )


# ══════════════════════════════════════════════════════════════════════
# MODEL 2: COMPLETE PIECEWISE-LINEAR MODEL  (CPWLM — Paper Eq. 14)
# ══════════════════════════════════════════════════════════════════════


def pwl_ode(
    t: float, y: np.ndarray, p: Parameters, steepness: float = 100
) -> list[float]:
    """CPWLM: same structure as CNM, Hill functions replaced by step functions."""
    rA, rB, pA, pB = y
    drA = p.mA * step_activation_smooth(pB, p.thetaB, steepness) - p.gammaA * rA
    drB = p.mB * step_inhibition_smooth(pA, p.thetaA, steepness) - p.gammaB * rB
    dpA = p.kPA * rA - p.deltaPA * pA
    dpB = p.kPB * rB - p.deltaPB * pB
    return [drA, drB, dpA, dpB]


def solve_pwl(p: Parameters, t_span=(0, 200), n_points=5000):
    """Integrate the CPWLM system using RK45."""
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    return solve_ivp(
        lambda t, y: pwl_ode(t, y, p),
        t_span,
        p.y0,
        t_eval=t_eval,
        method="RK45",
        dense_output=True,
    )


# ══════════════════════════════════════════════════════════════════════
# MODEL 3: DISCRETE-TIME MODEL  (Paper Eq. 23)
# ══════════════════════════════════════════════════════════════════════


def discrete_step(pA: float, pB: float, p: Parameters) -> tuple[float, float]:
    """One discrete time step (Paper Eq. 23)."""
    exp_dA = np.exp(-p.deltaPA)
    exp_dB = np.exp(-p.deltaPB)

    s_act = float(step_activation_hard(pB, p.thetaB))
    s_inh = float(step_inhibition_hard(pA, p.thetaA))

    pA_next = exp_dA * pA + (p.kA_prime / p.deltaPA) * (1 - exp_dA) * s_act
    pB_next = exp_dB * pB + (p.kB_prime / p.deltaPB) * (1 - exp_dB) * s_inh
    return pA_next, pB_next


def solve_discrete(p: Parameters, t_span=(0, 200)):
    """Iterate the discrete map (Paper Eq. 23)."""
    n_steps = int(t_span[1] - t_span[0]) + 1
    t = np.arange(n_steps)
    pA = np.zeros(n_steps)
    pB = np.zeros(n_steps)
    pA[0], pB[0] = p.y0[2], p.y0[3]

    for k in range(n_steps - 1):
        pA[k + 1], pB[k + 1] = discrete_step(pA[k], pB[k], p)

    rA = (p.mA / p.gammaA) * step_activation_hard(pB, p.thetaB)
    rB = (p.mB / p.gammaB) * step_inhibition_hard(pA, p.thetaA)

    return t, np.array([rA, rB, pA, pB])


# ══════════════════════════════════════════════════════════════════════
# BIFURCATION & STABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════


def steady_state_equations(y, params_tuple):
    """Residuals F(pA, pB) = 0 for the reduced steady-state system."""
    pA, pB = y
    mA, mB, gA, gB, kPA, kPB, thA, thB, nA, nB, dPA, dPB = params_tuple
    KA = (kPA * mA) / (dPA * gA)
    KB = (kPB * mB) / (dPB * gB)
    return [
        pA - KA * hill_activation(pB, thB, nB),
        pB - KB * hill_inhibition(pA, thA, nA),
    ]


def find_steady_states(params_tuple) -> list[np.ndarray]:
    """
    Find all equilibria by trying multiple initial guesses with fsolve.

    Returns a list of unique non-negative [pA*, pB*] solutions.
    """
    guesses = [
        [0.1, 0.1],
        [0.5, 0.5],
        [1.0, 1.0],
        [2.0, 0.1],
        [0.1, 2.0],
        [2.0, 2.0],
        [1.5, 0.5],
        [0.5, 1.5],
    ]
    results = []
    for g in guesses:
        try:
            x, info, ier, _ = fsolve(
                steady_state_equations, g, args=(params_tuple,), full_output=True
            )
            if ier == 1 and x[0] >= 0 and x[1] >= 0:
                if not any(np.allclose(x, r, atol=1e-4) for r in results):
                    results.append(x)
        except Exception:
            continue
    return results


def compute_jacobian(pA, pB, params_tuple) -> np.ndarray:
    """2×2 Jacobian of the reduced protein system at (pA, pB)."""
    mA, mB, gA, gB, kPA, kPB, thA, thB, nA, nB, dPA, dPB = params_tuple
    KA = (kPA * mA) / (dPA * gA)
    KB = (kPB * mB) / (dPB * gB)

    dh_act = (
        nB * thB**nB * pB ** (nB - 1) / (pB**nB + thB**nB) ** 2
        if nB > 0 and pB > 0
        else 0
    )
    dh_inh = (
        -nA * thA**nA * pA ** (nA - 1) / (pA**nA + thA**nA) ** 2
        if nA > 0 and pA > 0
        else 0
    )

    return np.array([[-dPA, KA * dh_act], [KB * dh_inh, -dPB]])


def classify_stability(pA, pB, params_tuple) -> tuple[str, np.ndarray]:
    """Classify equilibrium stability from 2D Jacobian eigenvalues."""
    eigs = np.linalg.eigvals(compute_jacobian(pA, pB, params_tuple))
    reals = np.real(eigs)
    if all(r < 0 for r in reals):
        return "stable", eigs
    elif all(r > 0 for r in reals):
        return "unstable", eigs
    return "saddle", eigs


# ── Full 4D CNM Jacobian & Hopf bifurcation analysis ────────────────


def compute_jacobian_4d(rA, rB, pA, pB, params_tuple) -> np.ndarray:
    """Full 4×4 Jacobian of the CNM at (rA, rB, pA, pB)."""
    mA, mB, gA, gB, kPA, kPB, thA, thB, nA, nB, dPA, dPB = params_tuple

    dh_act_dpB = (
        nB * thB**nB * pB ** (nB - 1) / (pB**nB + thB**nB) ** 2
        if nB > 0 and pB > 0
        else 0
    )
    dh_inh_dpA = (
        -nA * thA**nA * pA ** (nA - 1) / (pA**nA + thA**nA) ** 2
        if nA > 0 and pA > 0
        else 0
    )

    return np.array(
        [
            [-gA, 0, 0, mA * dh_act_dpB],
            [0, -gB, mB * dh_inh_dpA, 0],
            [kPA, 0, -dPA, 0],
            [0, kPB, 0, -dPB],
        ]
    )


def find_cnm_equilibria(params_tuple) -> list[np.ndarray]:
    """
    Find full 4D equilibria of the CNM.

    The protein steady states are identical to the 2D reduced system;
    the mRNA values follow from rA* = (mA/γA)·h⁺(pB*), rB* = (mB/γB)·h⁻(pA*).
    """
    mA, mB, gA, gB, kPA, kPB, thA, thB, nA, nB, dPA, dPB = params_tuple
    eq4d = []
    for ss in find_steady_states(params_tuple):
        pA_star, pB_star = ss
        rA_star = (mA / gA) * hill_activation(pB_star, thB, nB)
        rB_star = (mB / gB) * hill_inhibition(pA_star, thA, nA)
        eq4d.append(np.array([rA_star, rB_star, pA_star, pB_star]))
    return eq4d


def classify_stability_4d(eq, params_tuple) -> tuple[str, np.ndarray]:
    """Classify a 4D equilibrium from the full Jacobian eigenvalues."""
    rA, rB, pA, pB = eq
    J = compute_jacobian_4d(rA, rB, pA, pB, params_tuple)
    eigs = np.linalg.eigvals(J)
    reals = np.real(eigs)
    has_complex = any(abs(np.imag(e)) > 1e-8 for e in eigs)
    if all(r < 0 for r in reals):
        kind = "stable spiral" if has_complex else "stable node"
    elif all(r > 0 for r in reals):
        kind = "unstable spiral" if has_complex else "unstable node"
    elif any(r > 0 for r in reals) and any(r < 0 for r in reals):
        kind = "saddle"
    else:
        kind = "center / marginal"
    return kind, eigs


def compute_D_CNM(p: Parameters, pA_star: float, pB_star: float) -> float:
    """Paper Eq. 33: feedback strength parameter D_CNM at equilibrium."""
    num = (
        p.mA
        * p.mB
        * p.kPA
        * p.kPB
        * p.nA
        * p.nB
        * p.thetaA**p.nA
        * p.thetaB**p.nB
        * pA_star ** (p.nA - 1)
        * pB_star ** (p.nB - 1)
    )
    den = (p.thetaA**p.nA + pA_star**p.nA) ** 2 * (p.thetaB**p.nB + pB_star**p.nB) ** 2
    return num / den


def compute_D_Hopf(p: Parameters) -> float:
    """Paper Eq. 34: critical D value for Hopf bifurcation."""
    gA, gB = p.gammaA, p.gammaB
    dA, dB = p.deltaPA, p.deltaPB
    num = (gA + gB) * (gA + dA) * (gA + dB) * (gB + dA) * (gB + dB) * (dA + dB)
    den = (gA + gB + dA + dB) ** 2
    return num / den


def eigenvalues_from_D_CNM(D: float, p: Parameters) -> np.ndarray:
    """Roots of the CNM characteristic equation (Paper Eq. 32)."""
    gA, gB = p.gammaA, p.gammaB
    dA, dB = p.deltaPA, p.deltaPB
    # Expand the quartic
    c4 = 1.0
    c3 = gA + gB + dA + dB
    c2 = gA * gB + gA * dA + gA * dB + gB * dA + gB * dB + dA * dB
    c1 = gA * gB * dA + gA * gB * dB + gA * dA * dB + gB * dA * dB
    c0 = gA * gB * dA * dB + D
    return np.roots([c4, c3, c2, c1, c0])


def eigenvalues_from_D_SNM(D_SNM: float, p: Parameters) -> np.ndarray:
    """Roots of the SNM characteristic equation."""
    dA, dB = p.deltaPA, p.deltaPB
    return np.roots([1.0, dA + dB, dA * dB + D_SNM])


# ══════════════════════════════════════════════════════════════════════
# REGULATORY DOMAIN ANALYSIS  (PWL-specific)
# ══════════════════════════════════════════════════════════════════════


def analyze_regulatory_domains(p: Parameters):
    """Compute focal points in each of the 4 PWL regulatory domains."""
    K = p.kPA * p.mA / (p.deltaPA * p.gammaA)  # = ka'/δ = 2.35
    domains = {
        "D1 (pA<θ, pB<θ) — A off, B on": (0.0, K),
        "D2 (pA>θ, pB<θ) — A off, B off": (0.0, 0.0),
        "D3 (pA<θ, pB>θ) — A on,  B on": (K, K),
        "D4 (pA>θ, pB>θ) — A on,  B off": (K, 0.0),
    }
    return domains


# ══════════════════════════════════════════════════════════════════════
# PLOTTING — Generic helpers (DRY)
# ══════════════════════════════════════════════════════════════════════


def plot_timeseries(t, proteins, mrnas=None, *, title: str, save_path: str = None):
    """Generic time-series plot for one model."""
    pA, pB = proteins
    has_mrna = mrnas is not None

    if has_mrna:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        rA, rB = mrnas
        axes[0].plot(t, rA, color=COLOR_RA, label="mRNA A")
        axes[0].plot(t, rB, color=COLOR_RB, label="mRNA B")
        axes[0].set_ylabel("mRNA Concentration [M]")
        axes[0].set_title(f"{title} — mRNA")
        axes[0].legend(loc="best")
        axes[0].set_ylim(0, None)
        ax_p = axes[1]
    else:
        fig, ax_p = plt.subplots()

    ax_p.plot(t, pA, color=COLOR_PA, label="Protein A (GUARDIAN)")
    ax_p.plot(t, pB, color=COLOR_PB, label="Protein B (PROLIFERATOR)")
    ax_p.set_xlabel("Time [s]")
    ax_p.set_ylabel("Protein Concentration [M]")
    ax_p.set_title(f"{title} — Proteins")
    ax_p.legend(loc="best")
    ax_p.set_xlim(t[0], t[-1])
    ax_p.set_ylim(0, None)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_phase_portrait(
    pA, pB, *, title: str, thresholds=None, show_arrows=True, save_path: str = None
):
    """Generic phase portrait in the (pA, pB) plane."""
    fig, ax = plt.subplots()

    ax.plot(pA, pB, color="#9467bd", linewidth=1.5, label="Trajectory")
    ax.plot(pA[0], pB[0], "go", markersize=10, label="Start", zorder=5)
    ax.plot(pA[-1], pB[-1], "r*", markersize=15, label="End", zorder=5)

    if show_arrows:
        n_arrows = 10
        idxs = np.linspace(0, len(pA) - 2, n_arrows, dtype=int)
        for i in idxs:
            ax.annotate(
                "",
                xy=(pA[i + 1], pB[i + 1]),
                xytext=(pA[i], pB[i]),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
            )

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_vector_field(
    p: Parameters,
    reg_func_act,
    reg_func_inh,
    *,
    title: str,
    p_range=(0, 3),
    n_grid=20,
    save_path: str = None,
    **reg_kwargs,
):
    """Stream plot of the reduced 2D protein dynamics with nullclines."""
    fig, ax = plt.subplots()
    pA_arr = np.linspace(p_range[0] + 0.01, p_range[1], n_grid)
    pB_arr = np.linspace(p_range[0] + 0.01, p_range[1], n_grid)
    PA, PB = np.meshgrid(pA_arr, pB_arr)

    ka_prime = p.kPA * p.mA / p.gammaA  # = (kPA·mA/γA)
    kb_prime = p.kPB * p.mB / p.gammaB
    dpA = ka_prime * reg_func_act(PB, p.thetaB, **reg_kwargs) - p.deltaPA * PA
    dpB = kb_prime * reg_func_inh(PA, p.thetaA, **reg_kwargs) - p.deltaPB * PB

    # Constants for nullclines
    KA_full = ka_prime / p.deltaPA
    KB_full = kb_prime / p.deltaPB

    mag = np.sqrt(dpA**2 + dpB**2)
    mag[mag == 0] = 1

    strm = ax.streamplot(
        PA,
        PB,
        dpA,
        dpB,
        color=mag,
        cmap="viridis",
        density=1.5,
        linewidth=1,
        arrowsize=1.5,
    )

    # Nullclines
    pB_nc = np.linspace(0.01, p_range[1], 200)
    pA_nc_vals = KA_full * reg_func_act(pB_nc, p.thetaB, **reg_kwargs)
    pA_nc = np.linspace(0.01, p_range[1], 200)
    pB_nc_vals = KB_full * reg_func_inh(pA_nc, p.thetaA, **reg_kwargs)

    ax.plot(pA_nc_vals, pB_nc, "b--", linewidth=2, label="$\\dot{p}_A = 0$")
    ax.plot(pA_nc, pB_nc_vals, "r--", linewidth=2, label="$\\dot{p}_B = 0$")

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(p_range)
    ax.set_ylim(p_range)
    plt.colorbar(strm.lines, label="Vector Magnitude")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_multiple_trajectories(
    solver_fn,
    p: Parameters,
    initial_conditions=None,
    *,
    title: str,
    t_span=(0, 200),
    save_path: str = None,
):
    """Phase portrait with multiple ICs showing basin of attraction."""
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
        p_copy = Parameters()
        p_copy.y0 = list(y0)
        t, y = solver_fn(p_copy, t_span)
        ax.plot(
            y[2],
            y[3],
            color=colors[i],
            linewidth=1.5,
            label=f"IC: $p_A$={y0[2]:.1f}, $p_B$={y0[3]:.1f}",
        )
        ax.plot(y[2][0], y[3][0], "o", color=colors[i], markersize=8)
        ax.plot(y[2][-1], y[3][-1], "*", color=colors[i], markersize=12)

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════
# PLOTTING — Regulatory function comparison
# ══════════════════════════════════════════════════════════════════════


def plot_regulatory_functions(p: Parameters, save_path: str = None):
    """Compare Hill functions (n = 1, 2, 3, 5, 10) with the PWL step limit."""
    conc = np.linspace(0, 1.5, 500)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for n in [1, 2, 3, 5, 10]:
        axes[0].plot(conc, hill_activation(conc, p.thetaA, n), label=f"n = {n}")
        axes[1].plot(conc, hill_inhibition(conc, p.thetaB, n), label=f"n = {n}")

    # PWL limit
    s_act = step_activation_smooth(conc, p.thetaA)
    s_inh = step_inhibition_smooth(conc, p.thetaB)
    axes[0].plot(conc, s_act, "k--", linewidth=2, label="PWL (n→∞)")
    axes[1].plot(conc, s_inh, "k--", linewidth=2, label="PWL (n→∞)")

    for ax, lbl in zip(axes, ["Activation $h^+$", "Inhibition $h^-$"]):
        ax.set_xlabel("Protein Concentration [M]")
        ax.set_ylabel("Regulation Level")
        ax.set_title(lbl)
        ax.legend()
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════
# PLOTTING — Model comparison
# ══════════════════════════════════════════════════════════════════════


def plot_comparison_timeseries(sols: dict, save_path: str = None):
    """Overlay protein time series from all three models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, idx, name in [
        (axes[0], 2, "Protein A (GUARDIAN)"),
        (axes[1], 3, "Protein B (PROLIFERATOR)"),
    ]:
        ax.plot(
            sols["cnm"].t,
            sols["cnm"].y[idx],
            color=COLOR_CNM,
            linewidth=2,
            label="CNM (Hill)",
        )
        ax.plot(
            sols["pwl"].t,
            sols["pwl"].y[idx],
            color=COLOR_PWL,
            linewidth=2,
            ls="--",
            label="PWL (Step)",
        )
        t_d, y_d = sols["discrete"]
        ax.plot(
            t_d,
            y_d[idx],
            color=COLOR_DIS,
            linewidth=2,
            ls=":",
            label="Discrete (Eq. 23)",
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Concentration [M]")
        ax.set_title(name)
        ax.legend()
        ax.set_xlim(0, 50)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_comparison_phase(sols: dict, save_path: str = None):
    """Overlay phase portraits from all three models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        sols["cnm"].y[2],
        sols["cnm"].y[3],
        color=COLOR_CNM,
        linewidth=2,
        label="CNM (Hill)",
    )
    ax.plot(
        sols["pwl"].y[2],
        sols["pwl"].y[3],
        color=COLOR_PWL,
        linewidth=2,
        ls="--",
        label="PWL (Step)",
    )
    t_d, y_d = sols["discrete"]
    ax.plot(
        y_d[2], y_d[3], color=COLOR_DIS, linewidth=2, ls=":", label="Discrete (Eq. 23)"
    )

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title("Phase Portrait Comparison: CNM vs PWL vs Discrete")
    ax.legend(loc="best")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_comparison_phase_side_by_side(sols: dict, save_path: str = None):
    """Side-by-side phase portraits for the three models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = ["CNM (Hill Functions)", "PWL (Step Functions)", "Discrete (Paper Eq. 23)"]
    colors = [COLOR_CNM, COLOR_PWL, COLOR_DIS]

    data_list = [
        (sols["cnm"].y[2], sols["cnm"].y[3]),
        (sols["pwl"].y[2], sols["pwl"].y[3]),
        (sols["discrete"][1][2], sols["discrete"][1][3]),
    ]

    for ax, (pA, pB), lbl, col in zip(axes, data_list, labels, colors):
        ax.plot(pA, pB, color=col, linewidth=2)
        ax.plot(pA[0], pB[0], "go", markersize=10, label="Start")
        ax.plot(pA[-1], pB[-1], "r*", markersize=15, label="End")
        ax.set_xlabel("Protein A [M]")
        ax.set_ylabel("Protein B [M]")
        ax.set_title(lbl)
        ax.legend()
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 1.5)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_comparison_summary(sols: dict, save_path: str = None):
    """6-panel summary figure: time series + phase + bar chart."""
    fig = plt.figure(figsize=(16, 12))

    sol_c, sol_p = sols["cnm"], sols["pwl"]
    t_d, y_d = sols["discrete"]

    # Helper to add three traces
    def _add_traces(ax, idx, ylabel, xlim=50):
        ax.plot(sol_c.t, sol_c.y[idx], color=COLOR_CNM, lw=2, label="CNM")
        ax.plot(sol_p.t, sol_p.y[idx], color=COLOR_PWL, lw=2, ls="--", label="PWL")
        ax.plot(t_d, y_d[idx], color=COLOR_DIS, lw=2, ls=":", label="Discrete")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_xlim(0, xlim)

    ax1 = fig.add_subplot(2, 3, 1)
    _add_traces(ax1, 2, "Concentration [M]")
    ax1.set_title("Protein A (GUARDIAN)")

    ax2 = fig.add_subplot(2, 3, 2)
    _add_traces(ax2, 3, "Concentration [M]")
    ax2.set_title("Protein B (PROLIFERATOR)")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(sol_c.y[2], sol_c.y[3], color=COLOR_CNM, lw=2, label="CNM")
    ax3.plot(sol_p.y[2], sol_p.y[3], color=COLOR_PWL, lw=2, ls="--", label="PWL")
    ax3.plot(y_d[2], y_d[3], color=COLOR_DIS, lw=2, ls=":", label="Discrete")
    ax3.set_xlabel("Protein A [M]")
    ax3.set_ylabel("Protein B [M]")
    ax3.set_title("Phase Portrait Overlay")
    ax3.legend()
    ax3.set_xlim(0, 2.5)
    ax3.set_ylim(0, 1.0)

    ax4 = fig.add_subplot(2, 3, 4)
    _add_traces(ax4, 0, "Concentration [M]")
    ax4.set_title("mRNA A")

    ax5 = fig.add_subplot(2, 3, 5)
    _add_traces(ax5, 1, "Concentration [M]")
    ax5.set_title("mRNA B")

    ax6 = fig.add_subplot(2, 3, 6)
    variables = ["$p_A$", "$p_B$", "$r_A$", "$r_B$"]
    x = np.arange(4)
    w = 0.25
    for i, (src, col, lbl) in enumerate(
        [
            (
                [sol_c.y[2][-1], sol_c.y[3][-1], sol_c.y[0][-1], sol_c.y[1][-1]],
                COLOR_CNM,
                "CNM",
            ),
            (
                [sol_p.y[2][-1], sol_p.y[3][-1], sol_p.y[0][-1], sol_p.y[1][-1]],
                COLOR_PWL,
                "PWL",
            ),
            ([y_d[2][-1], y_d[3][-1], y_d[0][-1], y_d[1][-1]], COLOR_DIS, "Discrete"),
        ]
    ):
        ax6.bar(x + (i - 1) * w, src, w, label=lbl, color=col)
    ax6.set_xlabel("Variable")
    ax6.set_ylabel("Steady-State Value [M]")
    ax6.set_title("Steady-State Comparison")
    ax6.set_xticks(x)
    ax6.set_xticklabels(variables)
    ax6.legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════
# PATIENT BETA — SDEVelo (Mechanism II: Splicing Sabotage)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class SDEVeloParameters:
    """Parameters for the SDEVelo model (Assignment Table 4)."""

    # Logistic transcription switch: α(t) = c / (1 + exp(b·(t − a)))
    aA: float = 1.0  # switch time Gene A [s]
    aB: float = 0.25  # switch time Gene B [s]
    bA: float = 0.0005  # logistic steepness Gene A
    bB: float = 0.0005  # logistic steepness Gene B
    cA: float = 2.0  # max transcription rate Gene A [M·s⁻¹]
    cB: float = 0.5  # max transcription rate Gene B [M·s⁻¹]
    # Splicing rates [s⁻¹]
    betaA: float = 2.35
    betaB: float = 2.35
    # mRNA degradation rates [s⁻¹]
    gammaA: float = 1.0
    gammaB: float = 1.0
    # Hill parameters (shared with ODE model)
    nA: int = 3
    nB: int = 3
    thetaA: float = 0.21  # [M]
    thetaB: float = 0.21  # [M]
    # Translation / protein degradation [s⁻¹]
    kPA: float = 1.0
    kPB: float = 1.0
    deltaPA: float = 1.0
    deltaPB: float = 1.0
    # Noise intensities [M·s⁻¹ᐟ²]
    sigma_u: tuple[float, float] = (0.05, 0.05)
    sigma_s: tuple[float, float] = (0.05, 0.05)
    # Initial conditions [M]  — all species start at 0.8
    y0_unspliced: tuple[float, float] = (0.8, 0.8)
    y0_spliced: tuple[float, float] = (0.8, 0.8)
    y0_protein: tuple[float, float] = (0.8, 0.8)


def solve_sdevelo(
    sp: SDEVeloParameters,
    dt: float = 0.01,
    t_max: float = 20.0,
    n_realizations: int = 50,
    seed: int = 42,
) -> dict:
    """Euler-Maruyama integration of the SDEVelo SDE system."""
    rng = np.random.default_rng(seed)
    time = np.arange(0, t_max, dt)
    n_steps = time.size

    a = np.array([sp.aA, sp.aB])
    b = np.array([sp.bA, sp.bB])
    c = np.array([sp.cA, sp.cB])
    beta = np.array([sp.betaA, sp.betaB])
    gamma = np.array([sp.gammaA, sp.gammaB])
    n = np.array([sp.nA, sp.nB])
    theta = np.array([sp.thetaA, sp.thetaB])
    k = np.array([sp.kPA, sp.kPB])
    delta = np.array([sp.deltaPA, sp.deltaPB])
    sig_u = np.array(sp.sigma_u)
    sig_s = np.array(sp.sigma_s)

    all_pA, all_pB = [], []
    all_uA, all_uB = [], []
    all_sA, all_sB = [], []

    for _ in range(n_realizations):
        P = np.zeros((n_steps, 2))
        U = np.zeros((n_steps, 2))
        S = np.zeros((n_steps, 2))
        P[0] = sp.y0_protein
        U[0] = sp.y0_unspliced
        S[0] = sp.y0_spliced

        for i in range(1, n_steps):
            # Logistic transcription rate (time-dependent, NOT regulated)
            alpha = c / (1 + np.exp(b * (time[i] - a)))

            # Mechanism II: regulation acts on splicing (β), not transcription (α)
            contribution = np.array(
                [
                    hill_activation(P[i - 1, 1], theta[1], n[1]),
                    hill_inhibition(P[i - 1, 0], theta[0], n[0]),
                ]
            )
            beta_prime = beta * contribution

            # Euler-Maruyama with independent noise per gene
            dW_u = np.sqrt(dt) * rng.normal(size=2)
            dW_s = np.sqrt(dt) * rng.normal(size=2)

            U[i] = np.clip(
                U[i - 1] + (alpha - beta_prime * U[i - 1]) * dt + sig_u * dW_u, 0, None
            )
            S[i] = np.clip(
                S[i - 1]
                + (beta_prime * U[i - 1] - gamma * S[i - 1]) * dt
                + sig_s * dW_s,
                0,
                None,
            )
            P[i] = np.clip(P[i - 1] + (k * S[i - 1] - delta * P[i - 1]) * dt, 0, None)

        all_pA.append(P[:, 0])
        all_pB.append(P[:, 1])
        all_uA.append(U[:, 0])
        all_uB.append(U[:, 1])
        all_sA.append(S[:, 0])
        all_sB.append(S[:, 1])

    def _stats(arr):
        arr = np.array(arr)
        return np.mean(arr, axis=0), np.std(arr, axis=0)

    pA_m, pA_s = _stats(all_pA)
    pB_m, pB_s = _stats(all_pB)
    uA_m, uA_s = _stats(all_uA)
    uB_m, uB_s = _stats(all_uB)
    sA_m, sA_s = _stats(all_sA)
    sB_m, sB_s = _stats(all_sB)

    return {
        "time": time,
        "pA_mean": pA_m,
        "pA_std": pA_s,
        "pB_mean": pB_m,
        "pB_std": pB_s,
        "uA_mean": uA_m,
        "uA_std": uA_s,
        "uB_mean": uB_m,
        "uB_std": uB_s,
        "sA_mean": sA_m,
        "sA_std": sA_s,
        "sB_mean": sB_m,
        "sB_std": sB_s,
    }


def plot_sdevelo_timeseries(sol: dict, save_path: str = None):
    """Full time series for all 6 SDEVelo state variables with ±1σ bands.
    Improved visual clarity and legend placement.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    t = sol["time"]

    traces = [
        ("uA_mean", "uA_std", "Unspliced A (pre-mRNA)", "#e6a800"),
        ("sA_mean", "sA_std", "Spliced A (mRNA)", "#cc5500"),
        ("pA_mean", "pA_std", "Protein A (GUARDIAN)", "#cc0000"),
        ("uB_mean", "uB_std", "Unspliced B (pre-mRNA)", "#0d8f8f"),
        ("sB_mean", "sB_std", "Spliced B (mRNA)", "#006644"),
        ("pB_mean", "pB_std", "Protein B (PROLIFERATOR)", "#0a4005"),
    ]

    for mean_key, std_key, label, color in traces:
        m, s = sol[mean_key], sol[std_key]

        # Confidence band (slightly more visible but still subtle)
        ax.fill_between(
            t,
            np.clip(m - s, 0, None),
            m + s,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )

        # Main line with white outline for strong separation
        (line,) = ax.plot(t, m, label=label, color=color, linewidth=3.2, zorder=3)

        # Add contour/outline effect for clarity
        line.set_path_effects([pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])

    # Labels and title
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Concentration [M]", fontsize=12)
    ax.set_title(
        "SDEVelo — Mechanism II (Splicing Sabotage): Patient Beta",
        fontsize=13,
        weight="bold",
    )

    # Subtle grid improves readability of dense traces
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)

    # Move legend outside plot area (right side)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_sdevelo_phase(sol: dict, save_path: str = None):
    """SDEVelo protein-only phase portrait."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sol["pA_mean"], sol["pB_mean"], color="#9467bd", linewidth=2)
    ax.plot(sol["pA_mean"][0], sol["pB_mean"][0], "go", ms=10, label="Start", zorder=5)
    ax.plot(sol["pA_mean"][-1], sol["pB_mean"][-1], "r*", ms=15, label="End", zorder=5)
    ax.set_xlabel("Protein A (GUARDIAN) [M]")
    ax.set_ylabel("Protein B (PROLIFERATOR) [M]")
    ax.set_title("SDEVelo Phase Portrait — Patient Beta (Mechanism II)")
    ax.legend(loc="best")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ── Champion Selection & Cross-Mechanism Comparison ──────────────────


def select_champion(sols: dict) -> str:
    """Compare models for Patient Alpha and select the champion."""
    sol_c = sols["cnm"]
    sol_p = sols["pwl"]
    t_d, y_d = sols["discrete"]


    # 1. Check for oscillations (limit cycle)
    pA_late_cnm = sol_c.y[2][sol_c.t > 150]
    amp_cnm = np.max(pA_late_cnm) - np.min(pA_late_cnm) if len(pA_late_cnm) > 0 else 0

    pA_late_pwl = sol_p.y[2][sol_p.t > 150]
    amp_pwl = np.max(pA_late_pwl) - np.min(pA_late_pwl) if len(pA_late_pwl) > 0 else 0

    amp_dis = np.max(y_d[2][-50:]) - np.min(y_d[2][-50:]) if len(y_d[2]) > 50 else 0

    models = {
        "CNM": {
            "oscillates": amp_cnm > 0.01,
            "amplitude": amp_cnm,
            "pA_final": sol_c.y[2][-1],
            "pB_final": sol_c.y[3][-1],
            "n_vars": 4,
            "transcription_fn": "Hill (smooth, cooperative)",
            "assumptions": "None — full nonlinear dynamics",
        },
        "PWL": {
            "oscillates": amp_pwl > 0.01,
            "amplitude": amp_pwl,
            "pA_final": sol_p.y[2][-1],
            "pB_final": sol_p.y[3][-1],
            "n_vars": 4,
            "transcription_fn": "Step (piecewise-linear limit)",
            "assumptions": "n → ∞ (binary on/off regulation)",
        },
        "Discrete": {
            "oscillates": amp_dis > 0.01,
            "amplitude": amp_dis,
            "pA_final": y_d[2][-1],
            "pB_final": y_d[3][-1],
            "n_vars": 2,
            "transcription_fn": "Hard step (binary)",
            "assumptions": "Quasi-steady-state mRNA (γ >> δ), requires γ/δ >> 1",
        },
    }

    return "cnm"


def plot_mechanism_comparison(
    sol_alpha: object,
    sol_beta: dict,
    save_path: str = None,
):
    """
    Task 3: Side-by-side protein phase portraits for Patient Alpha (ODE/CNM,
    Mechanism I) and Patient Beta (SDEVelo, Mechanism II).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Patient Alpha — CNM phase portrait
    ax = axes[0]
    ax.plot(sol_alpha.y[2], sol_alpha.y[3], color=COLOR_CNM, linewidth=2)
    ax.plot(sol_alpha.y[2][0], sol_alpha.y[3][0], "go", ms=10, label="Start", zorder=5)
    ax.plot(sol_alpha.y[2][-1], sol_alpha.y[3][-1], "r*", ms=15, label="End", zorder=5)
    ax.set_xlabel("Protein A (GUARDIAN) [M]")
    ax.set_ylabel("Protein B (PROLIFERATOR) [M]")
    ax.set_title("Patient Alpha — Mechanism I\n(Transcriptional Hijack, CNM)")
    ax.legend(loc="best")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    # Patient Beta — SDEVelo phase portrait
    ax = axes[1]
    ax.plot(sol_beta["pA_mean"], sol_beta["pB_mean"], color="#9467bd", linewidth=2)
    ax.plot(
        sol_beta["pA_mean"][0],
        sol_beta["pB_mean"][0],
        "go",
        ms=10,
        label="Start",
        zorder=5,
    )
    ax.plot(
        sol_beta["pA_mean"][-1],
        sol_beta["pB_mean"][-1],
        "r*",
        ms=15,
        label="End",
        zorder=5,
    )
    ax.set_xlabel("Protein A (GUARDIAN) [M]")
    ax.set_ylabel("Protein B (PROLIFERATOR) [M]")
    ax.set_title("Patient Beta — Mechanism II\n(Splicing Sabotage, SDEVelo)")
    ax.legend(loc="best")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════
# BONUS — Downstream Metabolic Effects (Lotka-Volterra)
# ══════════════════════════════════════════════════════════════════════


def lotka_volterra_ode(t, y, params):
    """Lotka-Volterra ODE: dR/dt = αR − βRE, dE/dt = −γE + δRE."""
    R, E = y
    alpha, beta, gamma, delta = params
    dR = alpha * R - beta * R * E
    dE = -gamma * E + delta * R * E
    return [dR, dE]


def lotka_volterra_stability_analysis(alpha, beta, gamma, delta):
    """Stability analysis of the Lotka-Volterra system."""

    # FP2: coexistence
    R_star = gamma / delta
    E_star = alpha / beta
    J2 = np.array(
        [
            [alpha - beta * E_star, -beta * R_star],
            [delta * E_star, -gamma + delta * R_star],
        ]
    )
    eigs2 = np.linalg.eigvals(J2)
    return (R_star, E_star), eigs2


def plot_lotka_volterra_timeseries(params, y0, t_max=20.0, save_path=None):
    """Time series for the Lotka-Volterra system."""
    t_eval = np.linspace(0, t_max, 2000)
    sol = solve_ivp(
        lotka_volterra_ode,
        (0, t_max),
        y0,
        args=(params,),
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sol.t, sol.y[0], color="#e74c3c", linewidth=2, label="R (Resource)")
    ax.plot(sol.t, sol.y[1], color="#2980b9", linewidth=2, label="E (Enzyme)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Concentration [M]")
    ax.set_title("Lotka-Volterra: Metabolic Resource vs Growth Enzyme")
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig, sol


def plot_lotka_volterra_phase(sol, fp, params, save_path=None):
    """Phase portrait for the Lotka-Volterra system."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(sol.y[0], sol.y[1], color="#8e44ad", linewidth=2, label="Trajectory")
    ax.plot(sol.y[0][0], sol.y[1][0], "go", ms=10, label="Start", zorder=5)
    ax.plot(0, 0, "bs", ms=10, label=f"FP1 (0, 0) — saddle", zorder=5)
    ax.plot(
        fp[0],
        fp[1],
        "r^",
        ms=12,
        label=f"FP2 ({fp[0]:.2f}, {fp[1]:.2f}) — center",
        zorder=5,
    )
    ax.set_xlabel("R (Resource) [M]")
    ax.set_ylabel("E (Enzyme) [M]")
    ax.set_title("Lotka-Volterra: Phase Portrait")
    ax.legend(loc="best")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_lotka_volterra_stream(params, fp, save_path=None):
    """Stream plot with nullclines and equilibrium points."""
    alpha, beta, gamma, delta = params

    fig, ax = plt.subplots(figsize=(8, 7))

    # Vector field
    R_grid, E_grid = np.meshgrid(np.linspace(-1, 4, 50), np.linspace(-1, 4, 50))
    dR = alpha * R_grid - beta * R_grid * E_grid
    dE = -gamma * E_grid + delta * R_grid * E_grid
    speed = np.sqrt(dR**2 + dE**2)
    ax.streamplot(
        R_grid[0],
        E_grid[:, 0],
        dR,
        dE,
        color=speed,
        cmap="coolwarm",
        density=1.5,
        linewidth=0.8,
        arrowsize=1.2,
    )

    # Nullclines
    R_vals = np.linspace(-1, 4, 200)
    # dR/dt = 0 → E = α/β (horizontal line)
    ax.axhline(
        y=alpha / beta,
        color="#e74c3c",
        ls="--",
        linewidth=2,
        label=f"$\\frac{{dR}}{{dt}} = 0$ nullcline: $E = \\frac{{\\alpha}}{{\\beta}}$ = {alpha/beta:.2f}",
    )
    # dR/dt = 0 → R = 0 (horizontal line)
    ax.axhline(
        y=0,
        color="#e74c3c",
        ls="--",
        linewidth=2,
        label=f"$\\frac{{dR}}{{dt}} = 0$ nullcline: $E = 0$",
    )
    # dE/dt = 0 → R = γ/δ (vertical line)
    ax.axvline(
        x=gamma / delta,
        color="#2980b9",
        ls="--",
        linewidth=2,
        label=f"$\\frac{{dE}}{{dt}} = 0$ nullcline: $R = \\frac{{\\gamma}}{{\\delta}}$ = {gamma/delta:.2f}",
    )
    # dE/dt = 0 → E = 0 (vertical line)
    ax.axvline(
        x=0,
        color="#2980b9",
        ls="--",
        linewidth=2,
        label=f"$\\frac{{dE}}{{dt}} = 0$ nullcline: $R = 0$",
    )

    # Fixed points
    ax.plot(0, 0, "bs", ms=12, label="FP1 (0, 0) — saddle", zorder=5)
    ax.plot(
        fp[0],
        fp[1],
        "r^",
        ms=14,
        label=f"FP2 ({fp[0]:.2f}, {fp[1]:.2f}) — center",
        zorder=5,
    )

    ax.set_xlabel("R (Resource) [M]")
    ax.set_ylabel("E (Enzyme) [M]")
    ax.set_title("Lotka-Volterra: Stream Plot with Nullclines")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# PLOTTING — Bifurcation diagrams
# ══════════════════════════════════════════════════════════════════════


def plot_bifurcation(
    param_name: str,
    param_values: np.ndarray,
    base_params: Parameters,
    *,
    param_label: str,
    save_path: str = None,
):
    """Generic 1D bifurcation diagram varying one parameter."""
    pA_s, pB_s, v_s = [], [], []
    pA_u, pB_u, v_u = [], [], []

    for val in param_values:
        kw = {f"{param_name}_override": val}
        pt = base_params.as_tuple(**kw)
        for ss in find_steady_states(pt):
            stab, _ = classify_stability(ss[0], ss[1], pt)
            if stab == "stable":
                pA_s.append(ss[0])
                pB_s.append(ss[1])
                v_s.append(val)
            else:
                pA_u.append(ss[0])
                pB_u.append(ss[1])
                v_u.append(val)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data_s, data_u, ylabel in [
        (axes[0], pA_s, pA_u, "Protein A Steady-State [M]"),
        (axes[1], pB_s, pB_u, "Protein B Steady-State [M]"),
    ]:
        if v_s:
            ax.scatter(v_s, data_s, c="blue", s=20, label="Stable")
        if v_u:
            ax.scatter(v_u, data_u, c="red", s=20, marker="x", label="Unstable")
        ax.axhline(
            y=base_params.thetaA,
            color="gray",
            ls="--",
            alpha=0.5,
            label=f"θ = {base_params.thetaA}",
        )
        ax.set_xlabel(param_label)
        ax.set_ylabel(ylabel)
        ax.legend()
    axes[0].set_title(f"Protein A vs {param_label}")
    axes[1].set_title(f"Protein B vs {param_label}")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_nullclines_with_equilibria(p: Parameters, save_path: str = None):
    """Nullclines and equilibrium points with stability classification."""
    pt = p.as_tuple()
    KA = (p.kPA * p.mA) / (p.deltaPA * p.gammaA)
    KB = (p.kPB * p.mB) / (p.deltaPB * p.gammaB)

    fig, ax = plt.subplots(figsize=(10, 8))

    pB_r = np.linspace(0.001, 3, 500)
    pA_r = np.linspace(0.001, 3, 500)
    ax.plot(
        KA * hill_activation(pB_r, p.thetaB, p.nB),
        pB_r,
        "b-",
        lw=2,
        label="$\\dot{p}_A = 0$",
    )
    ax.plot(
        pA_r,
        KB * hill_inhibition(pA_r, p.thetaA, p.nA),
        "r-",
        lw=2,
        label="$\\dot{p}_B = 0$",
    )

    for ss in find_steady_states(pt):
        stab, eigs = classify_stability(ss[0], ss[1], pt)
        marker = "go" if stab == "stable" else "ro"
        ax.plot(
            ss[0],
            ss[1],
            marker,
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=2,
            label=f"{stab}: ({ss[0]:.2f}, {ss[1]:.2f})",
            zorder=10,
        )

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title(f"Nullclines and Equilibria (n={p.nA}, θ={p.thetaA})")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_bifurcation_2d(p: Parameters, resolution=30, save_path: str = None):
    """
    2D phase diagram in (n, θ) space showing number of stable equilibria.
    """
    n_vals = np.linspace(1, 8, resolution)
    th_vals = np.linspace(0.1, 0.5, resolution)
    count = np.zeros((resolution, resolution))

    for i, n in enumerate(n_vals):
        for j, th in enumerate(th_vals):
            pt = p.as_tuple(n_override=n, theta_override=th)
            for ss in find_steady_states(pt):
                stab, _ = classify_stability(ss[0], ss[1], pt)
                if stab == "stable":
                    count[j, i] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        count, extent=[1, 8, 0.1, 0.5], origin="lower", aspect="auto", cmap="viridis"
    )
    ax.plot(3, 0.21, "r*", markersize=20, label="Our parameters")
    ax.set_xlabel("Hill Coefficient (n)")
    ax.set_ylabel("Threshold θ [M]")
    ax.set_title("2D Bifurcation Diagram: Stable Equilibria Count (Reduced 2D)")
    ax.legend(loc="upper right")
    plt.colorbar(im, label="# Stable Equilibria")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ── Full 4D CNM Hopf Bifurcation Plots ─────────────────────────────


def plot_hopf_bifurcation(
    param_name: str,
    param_values: np.ndarray,
    base_params: Parameters,
    *,
    param_label: str,
    save_path: str = None,
):
    """1D Hopf bifurcation diagram for the full 4D CNM."""
    vals_list = []
    max_re_list = []
    pA_stable, pB_stable, v_stable = [], [], []
    pA_unstable, pB_unstable, v_unstable = [], [], []
    hopf_points = []  # (param_value, pA*, pB*)

    prev_max_re = None
    for val in param_values:
        kw = {f"{param_name}_override": val}
        pt = base_params.as_tuple(**kw)
        for eq in find_cnm_equilibria(pt):
            stab, eigs = classify_stability_4d(eq, pt)
            max_re = max(np.real(eigs))
            vals_list.append(val)
            max_re_list.append(max_re)
            pA_star, pB_star = eq[2], eq[3]

            if max_re < 0:
                pA_stable.append(pA_star)
                pB_stable.append(pB_star)
                v_stable.append(val)
            else:
                pA_unstable.append(pA_star)
                pB_unstable.append(pB_star)
                v_unstable.append(val)

            # Detect zero crossing → Hopf
            if prev_max_re is not None and prev_max_re * max_re < 0:
                hopf_points.append((val, pA_star, pB_star))
            prev_max_re = max_re

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Panel 1: max Re(λ)
    ax1.scatter(
        vals_list,
        max_re_list,
        c=["blue" if r < 0 else "red" for r in max_re_list],
        s=15,
        zorder=3,
    )
    ax1.axhline(y=0, color="black", lw=1.5, ls="-")
    for hp in hopf_points:
        ax1.axvline(x=hp[0], color="green", ls="--", lw=2, alpha=0.7)
    ax1.set_ylabel("max Re(λ) [4D Jacobian]")
    ax1.set_title(f"CNM Hopf Bifurcation Analysis (Full 4D) vs {param_label}")
    ax1.grid(alpha=0.3)
    # Annotate regions
    ax1.fill_between(
        param_values,
        0,
        max(max_re_list) * 1.2 if max_re_list else 1,
        alpha=0.05,
        color="red",
        label="Unstable (limit cycle)",
    )
    ax1.fill_between(
        param_values,
        min(max_re_list) * 1.2 if max_re_list else -1,
        0,
        alpha=0.05,
        color="blue",
        label="Stable focus",
    )
    ax1.legend(loc="upper left")

    # Panel 2: pA steady state
    if v_stable:
        ax2.scatter(v_stable, pA_stable, c="blue", s=15, label="Stable (4D)")
    if v_unstable:
        ax2.scatter(
            v_unstable, pA_unstable, c="red", s=15, marker="x", label="Unstable (4D)"
        )
    for hp in hopf_points:
        ax2.axvline(x=hp[0], color="green", ls="--", lw=2, alpha=0.7, label="Hopf")
    ax2.set_ylabel("Protein A Steady-State [M]")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Panel 3: pB steady state
    if v_stable:
        ax3.scatter(v_stable, pB_stable, c="blue", s=15, label="Stable (4D)")
    if v_unstable:
        ax3.scatter(
            v_unstable, pB_unstable, c="red", s=15, marker="x", label="Unstable (4D)"
        )
    for hp in hopf_points:
        ax3.axvline(x=hp[0], color="green", ls="--", lw=2, alpha=0.7, label="Hopf")
    ax3.set_xlabel(param_label)
    ax3.set_ylabel("Protein B Steady-State [M]")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)


    return fig


def plot_eigenvalue_comparison(p: Parameters, save_path: str = None):
    """Side-by-side comparison of 2D vs 4D eigenvalue spectra at equilibrium."""
    pt = p.as_tuple()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2D eigenvalues
    for ss in find_steady_states(pt):
        _, eigs_2d = classify_stability(ss[0], ss[1], pt)
        axes[0].plot(
            np.real(eigs_2d),
            np.imag(eigs_2d),
            "bo",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )
        for e in eigs_2d:
            axes[0].annotate(
                f"  λ = {e:.3f}", (np.real(e), np.imag(e)), fontsize=9, color="blue"
            )

    # 4D eigenvalues
    for eq in find_cnm_equilibria(pt):
        stab4d, eigs_4d = classify_stability_4d(eq, pt)
        colors = ["red" if np.real(e) > 0 else "blue" for e in eigs_4d]
        for e, c in zip(eigs_4d, colors):
            axes[1].plot(
                np.real(e),
                np.imag(e),
                "o",
                color=c,
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=1.5,
            )
            axes[1].annotate(
                f"  λ = {e:.3f}", (np.real(e), np.imag(e)), fontsize=9, color=c
            )

    for ax, title in [
        (axes[0], "Reduced 2D (quasi-steady-state mRNA)"),
        (axes[1], f"Full 4D CNM → {stab4d}"),
    ]:
        ax.axhline(y=0, color="black", lw=0.8)
        ax.axvline(x=0, color="black", lw=0.8)
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle("Eigenvalue Spectrum: 2D Reduced vs 4D CNM", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_hopf_2d_phase(p: Parameters, resolution=40, save_path: str = None):
    """2D Hopf phase diagram in (n, θ) space for the full 4D CNM."""
    n_vals = np.linspace(1, 8, resolution)
    th_vals = np.linspace(0.05, 0.8, resolution)
    max_re = np.full((resolution, resolution), np.nan)

    for i, n in enumerate(n_vals):
        for j, th in enumerate(th_vals):
            pt = p.as_tuple(n_override=n, theta_override=th)
            for eq in find_cnm_equilibria(pt):
                _, eigs = classify_stability_4d(eq, pt)
                mr = max(np.real(eigs))
                if np.isnan(max_re[j, i]) or mr > max_re[j, i]:
                    max_re[j, i] = mr

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        max_re,
        extent=[1, 8, 0.05, 0.8],
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
    )
    # Hopf curve: contour at max_re = 0
    ax.contour(
        n_vals,
        th_vals,
        max_re,
        levels=[0],
        colors="black",
        linewidths=2.5,
        linestyles="-",
    )
    ax.plot(3, 0.21, "r*", markersize=20, label="Our parameters", zorder=10)
    ax.set_xlabel("Hill Coefficient (n)")
    ax.set_ylabel("Threshold θ [M]")
    ax.set_title("4D CNM Hopf Bifurcation: max Re(λ) in (n, θ) Space")
    ax.legend(loc="upper right")
    cb = plt.colorbar(im, label="max Re(λ)")
    cb.ax.axhline(y=0, color="black", lw=2)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_nullclines_with_4d_stability(p: Parameters, save_path: str = None):
    """Nullclines with equilibria classified by the full 4D Jacobian."""
    pt = p.as_tuple()
    KA = (p.kPA * p.mA) / (p.deltaPA * p.gammaA)
    KB = (p.kPB * p.mB) / (p.deltaPB * p.gammaB)

    fig, ax = plt.subplots(figsize=(10, 8))

    pB_r = np.linspace(0.001, 3, 500)
    pA_r = np.linspace(0.001, 3, 500)
    ax.plot(
        KA * hill_activation(pB_r, p.thetaB, p.nB),
        pB_r,
        "b-",
        lw=2,
        label="$\\dot{p}_A = 0$",
    )
    ax.plot(
        pA_r,
        KB * hill_inhibition(pA_r, p.thetaA, p.nA),
        "r-",
        lw=2,
        label="$\\dot{p}_B = 0$",
    )

    # Overlay CNM limit cycle trajectory
    sol = solve_cnm(p, t_span=(0, 200))
    # Use last 60% to show the settled limit cycle
    n_skip = int(0.4 * len(sol.t))
    ax.plot(
        sol.y[2][n_skip:],
        sol.y[3][n_skip:],
        color="#2ca02c",
        lw=1.5,
        alpha=0.6,
        label="CNM limit cycle",
    )

    for eq in find_cnm_equilibria(pt):
        stab, eigs = classify_stability_4d(eq, pt)
        max_re = max(np.real(eigs))
        color = "green" if max_re < 0 else "red"
        ax.plot(
            eq[2],
            eq[3],
            "o",
            color=color,
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=2,
            label=f"4D: {stab}\n  ({eq[2]:.2f}, {eq[3]:.2f})",
            zorder=10,
        )

    ax.set_xlabel("Protein A Concentration [M]")
    ax.set_ylabel("Protein B Concentration [M]")
    ax.set_title(f"CNM Nullclines + 4D Stability (n={p.nA}, θ={p.thetaA})")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_paper_figure6(p: Parameters, save_path: str = None):
    """Reproduce Paper Figure 6: eigenvalues vs D_CNM / D_SNM."""
    D_Hopf = compute_D_Hopf(p)

    # Compute our actual D_CNM at the equilibrium
    pt = p.as_tuple()
    eqs = find_steady_states(pt)
    D_actual = None
    if eqs:
        D_actual = compute_D_CNM(p, eqs[0][0], eqs[0][1])

    # Sweep D_CNM
    D_range = np.linspace(-2, 15, 500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ── Panel (a): CNM eigenvalues vs D_CNM ──
    for D in D_range:
        eigs = eigenvalues_from_D_CNM(D, p)
        for e in eigs:
            is_complex = abs(np.imag(e)) > 1e-8
            color = "red" if is_complex else "black"
            ax1.plot(D, np.real(e), ".", color=color, markersize=2)

    ax1.axhline(y=0, color="gray", lw=1, ls="-")
    ax1.axvline(
        x=D_Hopf, color="green", lw=2, ls="--", label=f"$D_{{Hopf}}$ = {D_Hopf:.1f}"
    )
    if D_actual is not None:
        ax1.axvline(
            x=D_actual,
            color="blue",
            lw=2,
            ls=":",
            label=f"Our $D_{{CNM}}$ = {D_actual:.2f}",
        )
    ax1.set_xlabel("$D_{CNM}$")
    ax1.set_ylabel("Eigenvalue (real part)")
    ax1.set_title("(a) CNM: Eigenvalues vs $D_{CNM}$ (Paper Fig. 6a)")
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3)
    # Add legend for color convention
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=".",
            color="black",
            ls="",
            markersize=8,
            label="Real eigenvalue",
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            color="red",
            ls="",
            markersize=8,
            label="Re(complex pair)",
        ),
    ]
    leg2 = ax1.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax1.add_artist(ax1.legend(loc="lower left"))

    # ── Panel (b): SNM eigenvalues vs D_SNM ──
    D_SNM_range = D_range / (p.gammaA * p.gammaB)
    for D_snm in D_SNM_range:
        eigs = eigenvalues_from_D_SNM(D_snm, p)
        for e in eigs:
            is_complex = abs(np.imag(e)) > 1e-8
            color = "red" if is_complex else "black"
            ax2.plot(D_snm, np.real(e), ".", color=color, markersize=2)

    ax2.axhline(y=0, color="gray", lw=1, ls="-")
    if D_actual is not None:
        D_snm_actual = D_actual / (p.gammaA * p.gammaB)
        ax2.axvline(
            x=D_snm_actual,
            color="blue",
            lw=2,
            ls=":",
            label=f"Our $D_{{SNM}}$ = {D_snm_actual:.2f}",
        )
    ax2.set_xlabel("$D_{SNM}$")
    ax2.set_ylabel("Eigenvalue (real part)")
    ax2.set_title("(b) SNM: Eigenvalues vs $D_{SNM}$ (Paper Fig. 6b)")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)
    # Annotate that Re stays constant
    ax2.annotate(
        "Re(λ) = −(δA+δB)/2 = −1\n(always stable → NO Hopf)",
        xy=(D_SNM_range[len(D_SNM_range) // 2], -1),
        xytext=(D_SNM_range[len(D_SNM_range) // 3], -0.3),
        fontsize=10,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    plt.tight_layout()
    _save(fig, save_path)


    return fig


# ══════════════════════════════════════════════════════════════════════
# SOLVER WRAPPERS  (uniform interface for multi-trajectory plots)
# ══════════════════════════════════════════════════════════════════════


def _solve_cnm_uniform(p: Parameters, t_span=(0, 200)):
    sol = solve_cnm(p, t_span=t_span)
    return sol.t, sol.y


def _solve_pwl_uniform(p: Parameters, t_span=(0, 200)):
    sol = solve_pwl(p, t_span=t_span)
    return sol.t, sol.y


def _solve_discrete_uniform(p: Parameters, t_span=(0, 200)):
    return solve_discrete(p, t_span=t_span)


# ══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════


def main():
    """Run all models, generate all plots, and print analysis."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    p = Parameters()

    # ── 0. Viterbi Classification ────────────────────────────────────

    mech_a, mech_b = classify_patients()

    # ── 0b. Regulatory functions ─────────────────────────────────────

    plot_regulatory_functions(p, save_path="regulatory_functions.eps")

    # ── 1. CNM ───────────────────────────────────────────────────────
    sol_cnm = solve_cnm(p, t_span=(0, 200), n_points=5000)


    plot_timeseries(
        sol_cnm.t,
        (sol_cnm.y[2], sol_cnm.y[3]),
        (sol_cnm.y[0], sol_cnm.y[1]),
        title="CNM",
        save_path="cnm_timeseries.eps",
    )
    plot_phase_portrait(
        sol_cnm.y[2],
        sol_cnm.y[3],
        title="CNM: Phase Portrait",
        thresholds=(p.thetaA, p.thetaB),
        save_path="cnm_phase_portrait.eps",
    )
    plot_vector_field(
        p,
        lambda pB, th, **kw: hill_activation(pB, th, p.nB),
        lambda pA, th, **kw: hill_inhibition(pA, th, p.nA),
        title="CNM: Vector Field & Nullclines",
        save_path="cnm_vector_field.eps",
    )
    plot_multiple_trajectories(
        _solve_cnm_uniform,
        p,
        title="CNM: Multiple Trajectories",
        save_path="cnm_multi_trajectories.eps",
    )

    # ── 2. PWL ───────────────────────────────────────────────────────
    sol_pwl = solve_pwl(p, t_span=(0, 200), n_points=5000)
    analyze_regulatory_domains(p)

    plot_timeseries(
        sol_pwl.t,
        (sol_pwl.y[2], sol_pwl.y[3]),
        (sol_pwl.y[0], sol_pwl.y[1]),
        title="PWL",
        save_path="pwl_timeseries.eps",
    )
    plot_phase_portrait(
        sol_pwl.y[2],
        sol_pwl.y[3],
        title="PWL: Phase Portrait",
        thresholds=(p.thetaA, p.thetaB),
        save_path="pwl_phase_portrait.eps",
    )
    plot_vector_field(
        p,
        lambda pB, th, **kw: step_activation_smooth(pB, th),
        lambda pA, th, **kw: step_inhibition_smooth(pA, th),
        title="PWL: Vector Field & Switching Surfaces",
        save_path="pwl_vector_field.eps",
    )
    plot_multiple_trajectories(
        _solve_pwl_uniform,
        p,
        title="PWL: Multiple Trajectories",
        save_path="pwl_multi_trajectories.eps",
    )

    # ── 3. Discrete ──────────────────────────────────────────────────


    t_d, y_d = solve_discrete(p, t_span=(0, 200))

    # Use step plot style for discrete model
    fig, ax = plt.subplots()
    ax.step(t_d, y_d[2], where="post", color=COLOR_PA, label="Protein A (GUARDIAN)")
    ax.step(t_d, y_d[3], where="post", color=COLOR_PB, label="Protein B (PROLIFERATOR)")
    ax.plot(t_d, y_d[2], "o", color=COLOR_PA, markersize=2, alpha=0.3)
    ax.plot(t_d, y_d[3], "o", color=COLOR_PB, markersize=2, alpha=0.3)
    ax.set_xlabel("Discrete Time Step (n)")
    ax.set_ylabel("Protein Concentration [M]")
    ax.set_title("Discrete Model (Paper Eq. 23): Protein Time Series")
    ax.legend(loc="best")
    ax.set_xlim(t_d[0], t_d[-1])
    ax.set_ylim(0, None)
    plt.tight_layout()
    _save(fig, "discrete_timeseries.eps")

    plot_phase_portrait(
        y_d[2],
        y_d[3],
        title="Discrete Model: Phase Portrait",
        thresholds=(p.thetaA, p.thetaB),
        save_path="discrete_phase_portrait.eps",
    )
    plot_multiple_trajectories(
        _solve_discrete_uniform,
        p,
        title="Discrete: Multiple Trajectories",
        save_path="discrete_multi_trajectories.eps",
    )

    # ── 4. Model Comparison ──────────────────────────────────────────

    sols = {"cnm": sol_cnm, "pwl": sol_pwl, "discrete": (t_d, y_d)}

    plot_comparison_timeseries(sols, save_path="comparison_timeseries.eps")
    plot_comparison_phase(sols, save_path="comparison_phase_portrait.eps")
    plot_comparison_phase_side_by_side(
        sols, save_path="comparison_phase_side_by_side.eps"
    )
    plot_comparison_summary(sols, save_path="comparison_summary.eps")

    # ── 5. Champion Selection ────────────────────────────────────────

    champion = select_champion(sols)

    # ── 6. Patient Beta — SDEVelo (Mechanism II) ─────────────────────

    sp = SDEVeloParameters()
    sol_beta = solve_sdevelo(sp, dt=0.01, t_max=20.0, n_realizations=50)

    plot_sdevelo_timeseries(sol_beta, save_path="sdevelo_timeseries.eps")
    plot_sdevelo_phase(sol_beta, save_path="sdevelo_phase_portrait.eps")

    # ── 7. Task 3 — Comparative Analysis & Diagnosis ─────────────────

    sol_alpha_champion = sols[champion]
    plot_mechanism_comparison(
        sol_alpha_champion,
        sol_beta,
        save_path="mechanism_comparison.eps",
    )

    # ── 8. Bifurcation Analysis ──────────────────────────────────────

    # ── 8a. Reduced 2D analysis (quasi-steady-state mRNA) ──
    plot_nullclines_with_equilibria(p, save_path="bifurcation_nullclines_2d.eps")

    plot_bifurcation(
        "n",
        np.linspace(1, 10, 50),
        p,
        param_label="Hill Coefficient (n)",
        save_path="bifurcation_hill_coeff_2d.eps",
    )

    plot_bifurcation(
        "theta",
        np.linspace(0.05, 1.5, 50),
        p,
        param_label="Threshold θ [M]",
        save_path="bifurcation_threshold_2d.eps",
    )

    plot_bifurcation(
        "m",
        np.linspace(0.5, 5.0, 50),
        p,
        param_label="Transcription Rate m [s⁻¹]",
        save_path="bifurcation_transcription_2d.eps",
    )

    plot_bifurcation_2d(p, save_path="bifurcation_2d_phase_reduced.eps")

    # ── 8b. Full 4D CNM analysis (Hopf bifurcation) ──

    plot_paper_figure6(p, save_path="paper_figure6_eigenvalues.eps")

    plot_eigenvalue_comparison(p, save_path="eigenvalue_2d_vs_4d.eps")

    plot_nullclines_with_4d_stability(p, save_path="bifurcation_nullclines_4d.eps")

    plot_hopf_bifurcation(
        "n",
        np.linspace(1, 10, 80),
        p,
        param_label="Hill Coefficient (n)",
        save_path="hopf_hill_coeff.eps",
    )

    plot_hopf_bifurcation(
        "theta",
        np.linspace(0.05, 1.5, 80),
        p,
        param_label="Threshold θ [M]",
        save_path="hopf_threshold.eps",
    )

    plot_hopf_bifurcation(
        "m",
        np.linspace(0.5, 5.0, 80),
        p,
        param_label="Transcription Rate m [s⁻¹]",
        save_path="hopf_transcription.eps",
    )

    plot_hopf_2d_phase(p, save_path="hopf_2d_phase.eps")

    # ── 9. Bonus — Lotka-Volterra Metabolic Model ────────────────────

    lv_params = (2.0, 1.1, 1.0, 0.9)  # α, β, γ, δ
    lv_y0 = [1.0, 0.5]  # R(0), E(0)


    fp, eigs = lotka_volterra_stability_analysis(*lv_params)

    _, lv_sol = plot_lotka_volterra_timeseries(
        lv_params, lv_y0, t_max=20.0, save_path="lotka_volterra_timeseries.eps"
    )
    plot_lotka_volterra_phase(
        lv_sol, fp, lv_params, save_path="lotka_volterra_phase.eps"
    )
    plot_lotka_volterra_stream(lv_params, fp, save_path="lotka_volterra_stream.eps")

    # ── Done ─────────────────────────────────────────────────────────

    plt.close("all")
    return sols, sol_beta


if __name__ == "__main__":
    sols = main()
