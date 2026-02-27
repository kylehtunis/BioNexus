"""
Assignment 2 — Part 1: Gene Regulatory Network Modeling (Patient Alpha)
=========================================================================
Computational Biology, University of Amsterdam

Patient Alpha: RNA-seq data "AGCGC" → Viterbi classifies as Exon →
Mechanism I (Transcriptional Hijack) → ODE-based gene regulation model.

This file implements three progressively simplified models from:
    Polynikis, Hogan & di Bernardo (2009) "Comparing different ODE
    modelling approaches for gene regulatory networks"

Network topology (Figure 1 of the assignment):
    • Gene A (GUARDIAN)      — tumor-suppressor protein
    • Gene B (PROLIFERATOR)  — oncogenic protein
    • Protein B ACTIVATES transcription of Gene A  (positive regulation)
    • Protein A INHIBITS  transcription of Gene B  (negative regulation)

Models implemented:
    1. CNM  — Complete Nonlinear Model           (Paper Eq. 9–10, 4 ODEs)
    2. CPWLM — Complete Piecewise-Linear Model   (Paper Eq. 14,   4 ODEs)
    3. Discrete — Exact exponential integration   (Paper Eq. 23,   2 maps)

All parameters from Assignment Table 5.
"""

import os
import warnings
from dataclasses import dataclass, field


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

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
        print(f"  ✓ {full}")


# ══════════════════════════════════════════════════════════════════════
# PARAMETERS  (Assignment Table 5 — shared by all three models)
# ══════════════════════════════════════════════════════════════════════
@dataclass
class Parameters:
    """
    Kinetic parameters for the activator-inhibitor gene regulatory network.

    All values come from Assignment Table 5 (equivalent to Polynikis Table 5,
    Mechanism I — transcriptional regulation).

    The network has four state variables in the full (CNM/PWL) formulation:
        rA, rB : mRNA concentrations   [M]
        pA, pB : protein concentrations [M]

    Under the quasi-steady-state mRNA assumption (γ >> δ) used by the
    discrete model, mRNA is eliminated and only pA, pB are tracked.
    """

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
        """
        Combined production constant ka' = (mA / γA) · kPA.

        Arises from the quasi-steady-state mRNA assumption (Paper Eq. 13):
        when mRNA degrades much faster than protein (γ >> δ), we set
        drA/dt ≈ 0 → rA* = (mA/γA)·f(pB), then substitute into the protein
        equation: dpA/dt = kPA·rA* − δPA·pA = ka'·f(pB) − δPA·pA.

        For our parameters: ka' = (2.35/1.0)·1.0 = 2.35
        """
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
#
# These functions model how the concentration of a transcription-factor
# protein regulates the rate of transcription of a target gene.
#
# In the GUARDIAN/PROLIFERATOR network:
#   • h⁺(pB, θB, nB)  — Protein B *activates* Gene A transcription
#   • h⁻(pA, θA, nA)  — Protein A *inhibits* Gene B transcription
#
# The three models use different functional forms for these regulators,
# representing a trade-off between biological realism and mathematical
# tractability (Polynikis et al. 2009, Section 2):
#
#   CNM:      Hill functions (smooth sigmoids, cooperativity n)
#   CPWLM:    Step functions (limiting case n → ∞)
#   Discrete: Hard step functions (binary on/off)


def hill_activation(p: np.ndarray, theta: float, n: int) -> np.ndarray:
    """
    Hill activation function  h⁺(p, θ, n) = pⁿ / (pⁿ + θⁿ)

    Models cooperative binding of a transcription-factor activator:
    when n molecules of protein bind the promoter cooperatively, the
    transcription rate follows a sigmoidal curve that is steeper for
    higher n.  At p = θ the function equals 0.5 (half-maximal activation).

    Paper Eq. 3 (Polynikis et al.)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.power(p, n) / (np.power(p, n) + np.power(theta, n))


def hill_inhibition(p: np.ndarray, theta: float, n: int) -> np.ndarray:
    """
    Hill inhibition function  h⁻(p, θ, n) = θⁿ / (pⁿ + θⁿ)

    The complement of activation: h⁻ = 1 − h⁺.  When the repressor
    protein concentration exceeds θ, transcription is strongly suppressed.

    Paper Eq. 4
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.power(theta, n) / (np.power(p, n) + np.power(theta, n))


def step_activation_smooth(
    p: np.ndarray, theta: float, steepness: float = 100
) -> np.ndarray:
    """
    Smooth approximation of the PWL step function for activation.

    s⁺(p, θ) ≈ 1 / (1 + exp(−k(p − θ)))

    In the limit n → ∞ the Hill function becomes a Heaviside step at θ.
    We use a logistic sigmoid with large steepness k to numerically
    approximate this discontinuous switch while keeping the ODE solver
    stable (Polynikis Section 3).
    """
    return 1.0 / (1.0 + np.exp(-steepness * (p - theta)))


def step_inhibition_smooth(
    p: np.ndarray, theta: float, steepness: float = 100
) -> np.ndarray:
    """Smooth step inhibition: s⁻ = 1 − s⁺."""
    return 1.0 - step_activation_smooth(p, theta, steepness)


def step_activation_hard(p, theta) -> np.ndarray:
    """
    Hard (discontinuous) step function for the discrete model.

    s⁺(p, θ) = { 0  if p < θ
               { 1  if p ≥ θ

    This is the exact PWL switching function from Paper Eq. 14.
    Used in the discrete-time iteration (Eq. 23) where the step
    is evaluated once per timestep, not continuously.
    """
    return np.where(np.asarray(p) > theta, 1.0, 0.0)


def step_inhibition_hard(p, theta) -> np.ndarray:
    """Hard step inhibition: s⁻ = 1 − s⁺."""
    return np.where(np.asarray(p) < theta, 1.0, 0.0)


# ══════════════════════════════════════════════════════════════════════
# MODEL 1: COMPLETE NONLINEAR MODEL  (CNM — Paper Eq. 9–10)
# ══════════════════════════════════════════════════════════════════════
#
# The CNM retains full biological detail: mRNA and protein are tracked
# separately for both genes, giving a 4-dimensional ODE system.
#
# This is the most faithful model but also the hardest to analyse
# mathematically.  With our parameters (n = 3, θ = 0.21) the full 4D
# system exhibits *sustained oscillations* (a limit cycle), a result
# discussed in Polynikis Section 5.  The oscillations arise because
# the mRNA and protein dynamics create a delayed negative feedback loop
# whose period depends on the ratio γ/δ.


def cnm_ode(t: float, y: np.ndarray, p: Parameters) -> list[float]:
    """
    CNM: four coupled ODEs for the activator-inhibitor network.

        ṙA = mA · h⁺(pB, θB, nB) − γA · rA      (Gene A transcription)
        ṙB = mB · h⁻(pA, θA, nA) − γB · rB      (Gene B transcription)
        ṗA = kPA · rA − δPA · pA                  (Protein A translation)
        ṗB = kPB · rB − δPB · pB                  (Protein B translation)

    Each mRNA equation has:
      • a *production* term regulated by the opposing protein
        (Protein B activates Gene A;  Protein A inhibits Gene B)
      • a *degradation* term proportional to current mRNA level (first-order)

    Each protein equation has:
      • a *translation* term proportional to its mRNA
      • a *degradation* term (first-order proteolysis / dilution)

    Paper Eq. 9–10 (Polynikis et al. 2009)
    """
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
#
# The PWL model replaces the smooth Hill functions with step functions,
# corresponding to the biological idealisation of an *all-or-nothing*
# switch at the threshold concentration θ:
#
#   • Below θ the gene is fully OFF (or fully ON for inhibition)
#   • Above θ the gene is fully ON  (or fully OFF)
#
# This makes the system piecewise-linear: within each "regulatory
# domain" (region of phase space where all step values are constant)
# the ODEs are linear and can be solved analytically.
#
# The phase space is divided into 4 regulatory domains by the switching
# surfaces pA = θA and pB = θB (Paper Section 3, Eq. 14).  The key
# advantage is that equilibria and their stability can be determined
# analytically in each domain.


def pwl_ode(
    t: float, y: np.ndarray, p: Parameters, steepness: float = 100
) -> list[float]:
    """
    CPWLM: same structure as CNM, Hill functions replaced by step functions.

        ṙA = mA · s⁺(pB, θB) − γA · rA
        ṙB = mB · s⁻(pA, θA) − γB · rB
        ṗA = kPA · rA − δPA · pA
        ṗB = kPB · rB − δPB · pB

    Paper Eq. 14
    """
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
#
# The discrete model applies TWO simplifications on top of the CPWLM:
#
# (A) Quasi-steady-state mRNA (Paper Eq. 11–13):
#     If mRNA degrades much faster than protein (γ >> δ), mRNA reaches
#     its steady-state almost instantaneously compared to the protein
#     time scale.  Setting ṙ = 0 eliminates the mRNA variables:
#         rA* = (mA/γA) · s⁺(pB, θB)
#     Substituting into the protein equations gives the Simplified PWL
#     Model (SPWLM, Paper Eq. 15):
#         ṗA = ka' · s⁺(pB, θB) − δPA · pA
#         ṗB = kb' · s⁻(pA, θA) − δPB · pB
#     where ka' = (mA/γA)·kPA is the combined production constant.
#
#     ⚠ For our parameters γA = δPA = 1.0, so the assumption γ >> δ is
#     only marginally satisfied — a key observation from Polynikis
#     Section 5: the SPWLM may miss oscillatory dynamics present in
#     the full 4D system.
#
# (B) Exact exponential integration (Paper Eq. 21–23):
#     Between timesteps, s⁺ and s⁻ are constant (they only depend on
#     concentrations at the *previous* step).  The linear ODE for each
#     protein can then be integrated exactly:
#
#         pA(n+1) = exp(−δA) · pA(n) + (ka'/δA)(1 − exp(−δA)) · s⁺(pB(n))
#         pB(n+1) = exp(−δB) · pB(n) + (kb'/δB)(1 − exp(−δB)) · s⁻(pA(n))
#
#     This is NOT forward Euler — it is the analytical solution of the
#     linear part, so it is unconditionally stable regardless of step size.
#     The exponential terms represent protein decay; the second terms
#     represent production that accumulates during one timestep.


def discrete_step(pA: float, pB: float, p: Parameters) -> tuple[float, float]:
    """
    One discrete time step (Paper Eq. 23).

    Computes (pA(n+1), pB(n+1)) from (pA(n), pB(n)) using exact
    exponential integration of the SPWLM.
    """
    exp_dA = np.exp(-p.deltaPA)
    exp_dB = np.exp(-p.deltaPB)

    s_act = float(step_activation_hard(pB, p.thetaB))
    s_inh = float(step_inhibition_hard(pA, p.thetaA))

    pA_next = exp_dA * pA + (p.kA_prime / p.deltaPA) * (1 - exp_dA) * s_act
    pB_next = exp_dB * pB + (p.kB_prime / p.deltaPB) * (1 - exp_dB) * s_inh
    return pA_next, pB_next


def solve_discrete(p: Parameters, t_span=(0, 200)):
    """
    Iterate the discrete map (Paper Eq. 23).

    Returns (t, y) where y has shape (4, n_steps) for compatibility
    with the continuous solvers — rows 0,1 are mRNA (reconstructed
    from quasi-steady-state), rows 2,3 are the iterated proteins.
    """
    n_steps = int(t_span[1] - t_span[0]) + 1
    t = np.arange(n_steps)
    pA = np.zeros(n_steps)
    pB = np.zeros(n_steps)
    pA[0], pB[0] = p.y0[2], p.y0[3]

    for k in range(n_steps - 1):
        pA[k + 1], pB[k + 1] = discrete_step(pA[k], pB[k], p)

    # Reconstruct mRNA from quasi-steady-state (Paper Eq. 11)
    rA = (p.mA / p.gammaA) * step_activation_hard(pB, p.thetaB)
    rB = (p.mB / p.gammaB) * step_inhibition_hard(pA, p.thetaA)

    return t, np.array([rA, rB, pA, pB])


# ══════════════════════════════════════════════════════════════════════
# BIFURCATION & STABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════
#
# The bifurcation analysis uses the *reduced 2-variable system*
# obtained under the quasi-steady-state mRNA assumption:
#
#     ṗA = KA · h⁺(pB) − δPA · pA
#     ṗB = KB · h⁻(pA) − δPB · pB
#
# where KA = (kPA·mA)/(δPA·γA), KB = (kPB·mB)/(δPB·γB).
#
# At steady state both derivatives vanish, giving two implicit curves
# ("nullclines") in the (pA, pB) plane.  Their intersections are the
# equilibria.  Stability is assessed via the eigenvalues of the
# 2×2 Jacobian matrix.
#
# ⚠ This reduced system always yields a *stable spiral* for our
# parameters (eigenvalues ≈ −1 ± 2.56i), whereas the full 4D CNM
# exhibits a limit cycle.  The discrepancy highlights the danger of
# the quasi-steady-state assumption when γ ≈ δ (Polynikis Section 5).


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
    """
    2×2 Jacobian of the reduced protein system at (pA, pB).

    J = ⎡ −δPA          KA · ∂h⁺/∂pB ⎤
        ⎣ KB · ∂h⁻/∂pA       −δPB    ⎦

    The off-diagonal entries encode the cross-regulation:
      • ∂h⁺/∂pB > 0 (Protein B activates Gene A — positive feedback arm)
      • ∂h⁻/∂pA < 0 (Protein A inhibits Gene B — negative feedback arm)
    """
    mA, mB, gA, gB, kPA, kPB, thA, thB, nA, nB, dPA, dPB = params_tuple
    KA = (kPA * mA) / (dPA * gA)
    KB = (kPB * mB) / (dPB * gB)

    # Derivative of Hill activation: d/dp [pⁿ/(pⁿ+θⁿ)] = n·θⁿ·p^(n-1) / (pⁿ+θⁿ)²
    dh_act = (
        nB * thB**nB * pB ** (nB - 1) / (pB**nB + thB**nB) ** 2
        if nB > 0 and pB > 0
        else 0
    )
    # Derivative of Hill inhibition: d/dp [θⁿ/(pⁿ+θⁿ)] = −n·θⁿ·p^(n-1) / (pⁿ+θⁿ)²
    dh_inh = (
        -nA * thA**nA * pA ** (nA - 1) / (pA**nA + thA**nA) ** 2
        if nA > 0 and pA > 0
        else 0
    )

    return np.array([[-dPA, KA * dh_act], [KB * dh_inh, -dPB]])


def classify_stability(pA, pB, params_tuple) -> tuple[str, np.ndarray]:
    """Classify equilibrium stability from Jacobian eigenvalues."""
    eigs = np.linalg.eigvals(compute_jacobian(pA, pB, params_tuple))
    reals = np.real(eigs)
    if all(r < 0 for r in reals):
        return "stable", eigs
    elif all(r > 0 for r in reals):
        return "unstable", eigs
    return "saddle", eigs


# ══════════════════════════════════════════════════════════════════════
# REGULATORY DOMAIN ANALYSIS  (PWL-specific)
# ══════════════════════════════════════════════════════════════════════
#
# The switching surfaces pA = θA and pB = θB divide the protein phase
# space into four rectangular regulatory domains (Polynikis Section 3).
# In each domain the step functions take constant values (0 or 1),
# so the ODE system is linear and has a unique "focal point" — the
# point the trajectory is attracted toward while it stays in that domain.
#
# Whether the focal point lies inside its own domain determines whether
# the system can reach a true equilibrium there.


def analyze_regulatory_domains(p: Parameters):
    """Compute focal points in each of the 4 PWL regulatory domains."""
    K = p.kPA * p.mA / (p.deltaPA * p.gammaA)  # = ka'/δ = 2.35
    domains = {
        "D1 (pA<θ, pB<θ) — A off, B on": (0.0, K),
        "D2 (pA>θ, pB<θ) — A off, B off": (0.0, 0.0),
        "D3 (pA<θ, pB>θ) — A on,  B on": (K, K),
        "D4 (pA>θ, pB>θ) — A on,  B off": (K, 0.0),
    }
    print("\n┌─ Regulatory Domain Analysis (CPWLM) ─────────────┐")
    for name, (fpA, fpB) in domains.items():
        # Check if focal point lies inside its domain
        if "pA<θ" in name:
            valid_pA = fpA < p.thetaA
        else:
            valid_pA = fpA > p.thetaA
        if "pB<θ" in name:
            valid_pB = fpB < p.thetaB
        else:
            valid_pB = fpB > p.thetaB
        inside = "✓ inside" if (valid_pA and valid_pB) else "✗ outside"
        print(f"│  {name}")
        print(f"│    Focal point: (pA*={fpA:.3f}, pB*={fpB:.3f})  [{inside}]")
    print("└───────────────────────────────────────────────────┘\n")


# ══════════════════════════════════════════════════════════════════════
# PLOTTING — Generic helpers (DRY)
# ══════════════════════════════════════════════════════════════════════


def plot_timeseries(t, proteins, mrnas=None, *, title: str, save_path: str = None):
    """
    Generic time-series plot for one model.

    Args:
        t: time array
        proteins: (pA, pB) arrays
        mrnas: optional (rA, rB) arrays — if None, only proteins plotted
        title: subplot title prefix
    """
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
    """
    Generic phase portrait in the (pA, pB) plane.

    Args:
        pA, pB: protein concentration arrays
        thresholds: optional (θA, θB) to draw switching surfaces
        show_arrows: annotate trajectory direction
    """
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

    if thresholds:
        ax.axvline(
            x=thresholds[0],
            color="blue",
            ls="--",
            alpha=0.5,
            label=f"$\\theta_A = {thresholds[0]}$ M",
        )
        ax.axhline(
            y=thresholds[1],
            color="orange",
            ls="--",
            alpha=0.5,
            label=f"$\\theta_B = {thresholds[1]}$ M",
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
    """
    Stream plot of the reduced 2D protein dynamics with nullclines.

    Under quasi-steady-state mRNA:
        ṗA ≈ (kPA·mA/γA) · f⁺(pB) − δPA · pA
        ṗB ≈ (kPB·mB/γB) · f⁻(pA) − δPB · pB
    """
    fig, ax = plt.subplots()
    pA_arr = np.linspace(p_range[0] + 0.01, p_range[1], n_grid)
    pB_arr = np.linspace(p_range[0] + 0.01, p_range[1], n_grid)
    PA, PB = np.meshgrid(pA_arr, pB_arr)

    # Under quasi-steady-state mRNA: dpA/dt = ka'·f(pB) − δPA·pA
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
    """
    Phase portrait with multiple ICs showing basin of attraction.

    Args:
        solver_fn: callable(p, y0) → (t, y) with y shape (4, N)
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

    ax.axvline(x=p.thetaA, color="gray", ls="--", alpha=0.5)
    ax.axhline(y=p.thetaB, color="gray", ls="--", alpha=0.5)
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
    """
    Compare Hill functions (n = 1, 2, 3, 5, 10) with the PWL step limit.

    Shows how increasing cooperativity (n) makes the Hill function
    approach a discontinuous switch.  At our n = 3, the response is
    already fairly steep but still smooth, unlike the PWL idealisation.
    """
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
        ax.axvline(x=p.thetaA, color="gray", ls=":", alpha=0.5)
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

    ax.axvline(x=0.21, color="gray", ls="--", alpha=0.5, label="$\\theta_A$")
    ax.axhline(y=0.21, color="gray", ls=":", alpha=0.5, label="$\\theta_B$")

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
        ax.axvline(x=0.21, color="gray", ls="--", alpha=0.5)
        ax.axhline(y=0.21, color="gray", ls=":", alpha=0.5)
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
    ax3.axvline(x=0.21, color="gray", ls="--", alpha=0.3)
    ax3.axhline(y=0.21, color="gray", ls="--", alpha=0.3)
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
    """
    Generic 1D bifurcation diagram varying one parameter.

    For each parameter value, finds all equilibria and classifies
    their stability.  Stable equilibria are shown as blue dots,
    unstable as red crosses.
    """
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
    """
    Nullclines and equilibrium points with stability classification.

    The pA-nullcline is the curve where ṗA = 0:  pA = KA · h⁺(pB)
    The pB-nullcline is the curve where ṗB = 0:  pB = KB · h⁻(pA)
    Their intersections are the equilibria of the reduced system.
    """
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
        print(f"  Equilibrium ({ss[0]:.4f}, {ss[1]:.4f}): {stab}, λ = {eigs}")

    ax.axvline(x=p.thetaA, color="gray", ls="--", alpha=0.5)
    ax.axhline(y=p.thetaB, color="gray", ls="--", alpha=0.5)
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
    ax.axvline(x=3, color="red", ls="--", lw=2, label="n = 3")
    ax.axhline(y=0.21, color="red", ls=":", lw=2, label="θ = 0.21 M")
    ax.plot(3, 0.21, "r*", markersize=20, label="Our parameters")
    ax.set_xlabel("Hill Coefficient (n)")
    ax.set_ylabel("Threshold θ [M]")
    ax.set_title("2D Bifurcation Diagram: Stable Equilibria Count")
    ax.legend(loc="upper right")
    plt.colorbar(im, label="# Stable Equilibria")
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
# STEADY-STATE REPORTING
# ══════════════════════════════════════════════════════════════════════


def print_steady_state(label: str, rA, rB, pA, pB, note=""):
    """Print final-time values (approximate steady state)."""
    print(
        f"\n┌─ {label} ─ Approximate Steady State {'(' + note + ')' if note else ''}─┐"
    )
    print(f"│  mRNA A (rA):    {rA:.4f} M")
    print(f"│  mRNA B (rB):    {rB:.4f} M")
    print(f"│  Protein A (pA): {pA:.4f} M")
    print(f"│  Protein B (pB): {pB:.4f} M")
    print(f"└{'─' * 50}┘")


def print_comparison_table(sols: dict):
    """Print side-by-side steady-state comparison."""
    sc, sp = sols["cnm"], sols["pwl"]
    _, yd = sols["discrete"]

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║                  STEADY-STATE COMPARISON                       ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(
        f"║ {'Variable':<14} {'CNM (Hill)':<14} {'PWL (Step)':<14} {'Discrete':<14}  ║"
    )
    print("╠══════════════════════════════════════════════════════════════════╣")
    for name, ci in [
        ("mRNA A [M]", 0),
        ("mRNA B [M]", 1),
        ("Protein A [M]", 2),
        ("Protein B [M]", 3),
    ]:
        print(
            f"║ {name:<14} {sc.y[ci][-1]:<14.4f} {sp.y[ci][-1]:<14.4f} "
            f"{yd[ci][-1]:<14.4f}  ║"
        )
    print("╠══════════════════════════════════════════════════════════════════╣")

    for lbl, pA, pB in [
        ("CNM", sc.y[2][-1], sc.y[3][-1]),
        ("PWL", sp.y[2][-1], sp.y[3][-1]),
        ("Discrete", yd[2][-1], yd[3][-1]),
    ]:
        dom = "GUARDIAN dominates" if pA > pB else "PROLIFERATOR dominates"
        fate = "tumor-suppressor active" if pA > 0.5 else "tumor-suppressor inactive"
        print(f"║ {lbl:<10} → {dom:<22} → {fate:<20} ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")


# ══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════


def main():
    """Run all models, generate all plots, and print analysis."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    p = Parameters()

    # ── 0. Regulatory functions ──────────────────────────────────────
    print("\n" + "=" * 66)
    print("  ASSIGNMENT 2 — Part 1: Gene Regulatory Network (Patient Alpha)")
    print("  Mechanism I: Transcriptional Regulation")
    print("  Parameters from Table 5")
    print("=" * 66)

    print("\n📊 Plotting regulatory functions...")
    plot_regulatory_functions(p, save_path="regulatory_functions.eps")

    # ── 1. CNM ───────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  MODEL 1: Complete Nonlinear Model (CNM)")
    print("─" * 50)
    sol_cnm = solve_cnm(p, t_span=(0, 200), n_points=5000)
    print_steady_state("CNM", *[sol_cnm.y[i][-1] for i in range(4)])

    # Check for oscillations (limit cycle in 4D system)
    pA_late = sol_cnm.y[2][sol_cnm.t > 150]
    amplitude = np.max(pA_late) - np.min(pA_late)
    if amplitude > 0.01:
        print(f"\n  ⚠  OSCILLATING — pA amplitude = {amplitude:.4f} M")
        print("     → Limit cycle in 4D system (Polynikis Section 5)")
        print("     → Biologically: cyclic competition between GUARDIAN/PROLIFERATOR")
    else:
        print("\n  ✅ Converged to stable equilibrium")

    print("\n  Generating CNM plots...")
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
    print("\n" + "─" * 50)
    print("  MODEL 2: Complete Piecewise-Linear Model (CPWLM)")
    print("─" * 50)
    sol_pwl = solve_pwl(p, t_span=(0, 200), n_points=5000)
    print_steady_state("PWL", *[sol_pwl.y[i][-1] for i in range(4)])
    analyze_regulatory_domains(p)

    print("  Generating PWL plots...")
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
    print("\n" + "─" * 50)
    print("  MODEL 3: Discrete-Time Model (Paper Eq. 23)")
    print("─" * 50)

    # Print derived constants
    coeff = (p.kA_prime / p.deltaPA) * (1 - np.exp(-p.deltaPA))
    print(f"  ka' = {p.kA_prime:.4f},  α = exp(−δ) = {p.alpha_A:.4f}")
    print(f"  Production coefficient = (ka'/δ)(1−α) = {coeff:.4f}")
    print(f"  Iteration: pA(n+1) = {p.alpha_A:.4f}·pA(n) + {coeff:.4f}·s⁺(pB(n))")

    t_d, y_d = solve_discrete(p, t_span=(0, 200))
    print_steady_state(
        "Discrete", *[y_d[i][-1] for i in range(4)], note="mRNA from quasi-steady-state"
    )

    print("\n  Generating Discrete plots...")
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
    print("\n" + "─" * 50)
    print("  MODEL COMPARISON")
    print("─" * 50)

    sols = {"cnm": sol_cnm, "pwl": sol_pwl, "discrete": (t_d, y_d)}
    print_comparison_table(sols)

    print("  Generating comparison plots...")
    plot_comparison_timeseries(sols, save_path="comparison_timeseries.eps")
    plot_comparison_phase(sols, save_path="comparison_phase_portrait.eps")
    plot_comparison_phase_side_by_side(
        sols, save_path="comparison_phase_side_by_side.eps"
    )
    plot_comparison_summary(sols, save_path="comparison_summary.eps")

    # ── 5. Bifurcation Analysis ──────────────────────────────────────
    print("\n" + "─" * 50)
    print("  BIFURCATION ANALYSIS")
    print("─" * 50)

    print("  Nullclines and equilibria...")
    plot_nullclines_with_equilibria(p, save_path="bifurcation_nullclines.eps")

    print("  Bifurcation vs Hill coefficient n...")
    plot_bifurcation(
        "n",
        np.linspace(1, 10, 50),
        p,
        param_label="Hill Coefficient (n)",
        save_path="bifurcation_hill_coeff.eps",
    )

    print("  Bifurcation vs threshold θ...")
    plot_bifurcation(
        "theta",
        np.linspace(0.05, 1.5, 50),
        p,
        param_label="Threshold θ [M]",
        save_path="bifurcation_threshold.eps",
    )

    print("  Bifurcation vs transcription rate m...")
    plot_bifurcation(
        "m",
        np.linspace(0.5, 5.0, 50),
        p,
        param_label="Transcription Rate m [s⁻¹]",
        save_path="bifurcation_transcription.eps",
    )

    print("  2D bifurcation phase diagram (n, θ)...")
    plot_bifurcation_2d(p, save_path="bifurcation_2d_phase.eps")

    # ── Done ─────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  All plots saved to ./output/")
    print("=" * 66)

    plt.close("all")
    return sols


if __name__ == "__main__":
    sols = main()
