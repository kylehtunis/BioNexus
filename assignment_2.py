import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
from scipy.integrate import solve_ivp

sns.set_theme()

# Viterbi
def viterbi(sequence, Initial, Emissions, Transitions, state=-1):
    if state < 0:
        V_0, backtrack_0 = viterbi(sequence, Initial, Emissions, Transitions, 0)
        V_1, backtrack_1 = viterbi(sequence, Initial, Emissions, Transitions, 1)
        if V_0 > V_1:
            return V_0, backtrack_0 + "E"
        else:
            return V_1, backtrack_1 + "I"
    else:
        emitted_value = sequence[-1]
        if sequence.size == 1:
            return Initial[state] * Emissions[state, emitted_value], f""
        V_0, backtrack_0 = viterbi(sequence[:-1], Initial, Emissions, Transitions, 0)
        V_0 *= Transitions[state, 0] * Emissions[state, emitted_value]
        V_1, backtrack_1 = viterbi(sequence[:-1], Initial, Emissions, Transitions, 1)
        V_1 *= Transitions[state, 1] * Emissions[state, emitted_value]
        if V_0 > V_1:
            return V_0, backtrack_0 + "E"
        else:
            return V_1, backtrack_1 + "I"

def ode(t, state):
    ma = 2.35
    mb = 2.35
    gama = 1
    gamb = 1
    kpa = 1
    kpb = 1
    thea = .21
    theb = .21
    na = 3
    nb = 3
    delpa = 1
    delpb = 1

    pa, pb, ra, rb = state
    dpa = kpa * ra - delpa * pa
    dpb = kpb * rb - delpb * pb
    dra = ma * hill(pb, theb, nb) - gama * ra
    drb = mb * (1 - hill(pa, thea, na)) - gamb * rb
    return [dpa, dpb, dra, drb]

def patient_alpha_ODE(timestep, t_max):
    solution = sp.integrate.solve_ivp(ode, (0, t_max), [.8, .8, .8, .8])
    protein_a, protein_b, mrna_a, mrna_b = solution.y
    time = solution.t
    
    plt.plot(time, protein_a, label="Pa")
    plt.plot(time, protein_b, label="Pb")
    plt.plot(time, mrna_a, label="ra")
    plt.plot(time, mrna_b, label="rb")
    plt.legend()
    plt.show()
    plt.plot(protein_a, protein_b)
    plt.show()

<<<<<<< HEAD
def hill(p, theta, n):
    return p**n / (p**n + theta**n)

def SDEVelo(timestep, t_max, ngens=50):
    rng = np.random.default_rng(42)
    time = np.arange(0, t_max, timestep)
    a = np.array([1., .25])
    b = np.array([.0005, .0005])
    c = np.array([2., .5])
    beta = np.array([2.35, 2.35])
    gamma = np.array([1., 1.])
    n = np.array([3, 3])
    theta = np.array([.21, .21])
    k = np.array([1., 1.])
    m = np.array([2.35, 2.35])
    delta = np.array([1., 1.])
    sigma = np.array([
        [.05, .05],
        [.05, .05]
        ])
    M_0 = .8
    pa = []
    pb = []
    ua = []
    ub = []
    sa = []
    sb = []
    for gen in range(ngens):
        protein_a = np.zeros_like(time)
        protein_a[0] = M_0
        protein_b = np.zeros_like(time)
        protein_b[0] = M_0
        unspliced_a = np.zeros_like(time)
        unspliced_a[0] = M_0
        unspliced_b = np.zeros_like(time)
        unspliced_b[0] = M_0
        spliced_a = np.zeros_like(time)
        spliced_a[0] = M_0
        spliced_b = np.zeros_like(time)
        spliced_b[0] = M_0
        for idx in range(1, time.size):
            P = np.array([protein_a[idx - 1], protein_b[idx - 1]])
            S = np.array([spliced_a[idx-1], spliced_b[idx-1]])
            U = np.array([unspliced_a[idx-1], unspliced_b[idx-1]])
            alpha = c / (1 + np.exp(b) * (time[idx] - a))
            contribution = np.array([hill(P[1], theta[1], n[1]), 1 - hill(P[0], theta[0], n[0])])
            alpha_prime = alpha * contribution
            beta_prime = beta * contribution
            # print(f"P: {P}\nS: {S}\nU: {U}\nC: {contribution}")
            # protein_a[idx] = max(protein_a[idx-1] + timestep * (k[0] * spliced_a[idx - 1] - delta[0] * protein_a[idx - 1]), 0)
            # protein_b[idx] = max(protein_b[idx - 1] + timestep * (k[1] * spliced_b[idx - 1] - delta[1] * protein_b[idx - 1]), 0)
            protein_a[idx], protein_b[idx] = np.clip(P + timestep * (k * S - delta * P), 0, None)
            unspliced_a[idx], unspliced_b[idx] = np.clip(U + timestep * (alpha_prime - beta_prime* U) + sigma[0]*np.sqrt(timestep) * rng.normal(), 0, None)
            spliced_a[idx], spliced_b[idx] = np.clip(S + (beta_prime* U - gamma*S) * timestep + sigma[1] * np.sqrt(timestep) * rng.normal(), 0, None)
        pa.append(protein_a)
        pb.append(protein_b)
        ua.append(unspliced_a)
        ub.append(unspliced_b)
        sa.append(spliced_a)
        sb.append(spliced_b)
    
    pa = np.array(pa)
    pa_m = np.mean(pa, axis=0)
    pa_v = np.sqrt(np.var(pa, axis=0))
    pb = np.array(pb)
    pb_m = np.mean(pb, axis=0)
    pb_v = np.sqrt(np.var(pb, axis=0))

    ua = np.array(ua)
    ua_m = np.mean(ua, axis=0)
    ua_v = np.sqrt(np.var(ua, axis=0))
    ub = np.array(ub)
    ub_m = np.mean(ub, axis=0)
    ub_v = np.sqrt(np.var(ub, axis=0))

    sa = np.array(sa)
    sa_m = np.mean(sa, axis=0)
    sa_v = np.sqrt(np.var(sa, axis=0))
    sb = np.array(sb)
    sb_m = np.mean(sb, axis=0)
    sb_v = np.sqrt(np.var(sb, axis=0))

    plt.plot(time, ua_m, label="unspliced a", color='#ffc700')
    plt.fill_between(time, np.clip(ua_m - ua_v, a_min=0, a_max=None), ua_m + ua_v, color='#ffc700', alpha=.5)
    plt.plot(time, sa_m, label="spliced a", color="#ff8300")
    plt.plot(time, pa_m, label="protein a", color='#ff0000')
    plt.fill_between(time, np.clip(pa_m - pa_v, 0, a_max=None), pa_m + pa_v, color='#ff0000', alpha=.5)
    plt.fill_between(time, np.clip(sa_m - sa_v, 0, a_max=None), sa_m + sa_v, color='#ff8300', alpha=.5)
    plt.plot(time, ub_m, label="unspliced b", color='#10b7b7')
    plt.fill_between(time, np.clip(ub_m - ub_v, a_min=0, a_max=None), ub_m + ub_v, color='#10b7b7', alpha=.5)
    plt.plot(time, sb_m, label="spliced b", color="#008c63")
    plt.fill_between(time, np.clip(sb_m - sb_v, 0, a_max=None), sb_m + sb_v, color='#008c63', alpha=.5)
    plt.plot(time, pb_m, label="protein b", color="#105f07")
    plt.fill_between(time, np.clip(pb_m - pb_v, 0, a_max=None), pb_m + pb_v, color='#105f07', alpha=.5)
    plt.legend()
    plt.show()
    plt.plot(pa_m, pb_m)
    plt.show()




=======
def downstream_ode(t, initial, params):
    R, E = initial
    alpha, beta, gamma, delta = params
    dR_dt = alpha * R - beta * R * E
    dE_dt = -gamma * E + delta * R * E
    return (dR_dt, dE_dt)

def solve_and_plot(func, initial, params, t_max, timestep):
    time = np.linspace(0, t_max, int(t_max / timestep))
    sol = solve_ivp(func, (0, t_max), initial, args=(params,), t_eval=time)
    plt.plot(time, sol.y[0], label="R")
    plt.plot(time, sol.y[1], label="E")
    plt.legend()
    plt.show()
    return sol

# ODE 
>>>>>>> 581fa085a5cffa216e10f2619ef9bd9d32a91c13
def main():
    Initial = np.array([.5, .5])
    Emissions = np.array([
        [.25, .25, .25, .25],
        [.4, .4, .05, .15]
    ])
    Transitions = np.array([
        [.9, .1],
        [.2, .8]
    ])
    patient_alpha = np.array([0,2,3,2,3])
    patient_beta = np.array([0,1,1,0,1])
    print(viterbi(patient_alpha, Initial, Emissions, Transitions, state=-1))
    print(viterbi(patient_beta, Initial, Emissions, Transitions, state=-1))
    # patient_alpha_ODE(.1, 50)
    SDEVelo(.01, 20)

    # part 2 ode
    sol = solve_and_plot(downstream_ode, (1.0, 0.5), (2, 1.1, 1, 0.9), 20, 0.1)
    # part 2 phase plot
    plt.plot(sol.y[0], sol.y[1])
    plt.xlabel("R")
    plt.ylabel("E")
    plt.title("Phase Plane")
    plt.show()
    # part 2 stream plot
    E, R = np.mgrid[-1:3:.1, -1:3:.1]
    dR_dt, dE_dt = downstream_ode(0, (R, E), (2, 1.1, 1, 0.9))
    print(R[0], E[0])
    plt.streamplot(R[0], E[:,0], dR_dt, dE_dt)
    plt.xlabel("R")
    plt.ylabel("E")
    plt.plot(0, 0, 'bo', label="Fixed Point 1")
    plt.plot(10/9, 20/11, 'ro', label="Fixed Point 2")
    plt.title("Stream Plot")
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()