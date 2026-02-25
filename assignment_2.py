import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def patient_alpha_ODE(timestep, t_max):
    time = np.linspace(0, t_max, int(t_max / timestep))
    protein_a = np.zeros_like(time)
    protein_b = np.zeros_like(time)
    mrna_a = np.zeros_like(time)
    mrna_b = np.zeros_like(time)

    protein_a[0] = .8
    protein_b[0] = .8
    mrna_a[0] = .8
    mrna_b[0] = .8

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

    for idx in range(1, time.size):
        protein_a[idx] = protein_a[idx-1] +  timestep * (kpa * mrna_a[idx - 1] - delpa * protein_a[idx - 1])
        mrna_a[idx] = mrna_a[idx-1] + timestep * (ma * protein_b[idx - 1]**nb / (protein_b[idx-1]**nb + theb**nb) - gama * mrna_a[idx - 1])
        protein_b[idx] = protein_b[idx - 1] + timestep * (kpb * mrna_b[idx - 1] - delpb * protein_b[idx - 1])
        mrna_b[idx] = mrna_b[idx - 1] + timestep * (mb * thea**na / (thea**na + protein_a[idx-1]**thea) - gamb * mrna_b[idx-1])
    plt.plot(time, protein_a, label="Pa")
    plt.plot(time, protein_b, label="Pb")
    plt.plot(time, mrna_a, label="ra")
    plt.plot(time, mrna_b, label="rb")
    plt.legend()
    plt.show()
    plt.plot(protein_a, protein_b)
    plt.show()

# ODE 
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
    patient_alpha_ODE(.1, 50)

if __name__ == "__main__":
    main()