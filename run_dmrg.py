import numpy as np
import matplotlib.pyplot as plt
import scipy

import tensorNets.a_mps as a_mps
import tensorNets.d_dmrg as d_dmrg

from heisenbergModel import HeisenbergModel, HeisenbergModelNearestNeighbors, HeisenbergModelNoDecay

# function to run the DMRG algorithm

Sz = np.array([[1, 0.], [0., -1]])

def run_dmrg_simulation(L, J, xi, max_sweeps=20, tol=1e-9, chi_max=100):
    mps = a_mps.init_spinup_MPS(L)
    model = HeisenbergModel(L, J, xi)
    engine = d_dmrg.DMRGEngine(mps, model, chi_max=chi_max)

    # E_exact = tfi_exact.finite_gs_energy(L, J, g)

    E, E_prev = 0, 0

    for i in range(max_sweeps):
        engine.sweep()
        E_prev = E
        E = engine.calculate_energy()
        print("Energy:", E)
        # print("Energy error:", E - E_exact)
        if np.abs(E - E_prev) < tol:
            print(f"The system is converged after {i+1} sweeps!")
            break
    
    return model, mps, engine

def calculate_two_operator_expectation(psi, X, Y, i):
    """
    Calculate the expectation value of two operators on the MPS.
    <psi|X_i X_j|psi>
    where |psi> is the MPS.

    calculate for all j >= i
    
    """
    return_list = []

    # first calculate the left hand structure. its always the same so we can precompute and update it.
    theta_i = psi.get_theta1(i) 
    X_theta_i = np.tensordot(X, theta_i, axes=[1, 1])  # i [i*], vL [i] vR
    left_exp_struct = np.tensordot(theta_i.conj(), X_theta_i, [[0, 1], [1, 0]]) # [vL*] [i*] vR*, [i] [vL] vR

    for j in range(i, psi.L):
        if j == i:
            op  = np.tensordot(X, Y, [1, 0]) # i [i*], [i] i*
            op_theta = np.tensordot(op, theta_i, axes=[1, 1])  # i [i*], vL [i] vR
            exp_val = np.tensordot(theta_i.conj(), op_theta, [[0, 1, 2], [1, 0, 2]]) # [vL*] [i*] [vR*], [i] [vL] [vR]
            return_list.append(exp_val)
        else:
            tmp = np.tensordot(Y, psi.Bs[j], axes=[1,1]) # i [i*], vL [i] vR
            right_side = np.tensordot(psi.Bs[j].conj(), tmp, axes=[[1,2],[0,2]])  # vL* [i*] [vR*], [i] vL [vR]
            exp_val = np.tensordot(left_exp_struct, right_side, [[0, 1], [0, 1]]) # [vR*] [vR], [vL*] [vL]
            return_list.append(exp_val)
            # update the left structure
            left_exp_struct = np.tensordot(left_exp_struct, psi.Bs[j], axes=[1,0]) # vR* [vR] x [vL], i, vR
            left_exp_struct = np.tensordot(left_exp_struct, psi.Bs[j].conj(), axes=[[1,0],[1,0]]) # [vR*] [i] vR x [vL*] [i*] vR*
            left_exp_struct = np.transpose(left_exp_struct, (1,0)) # vR* vR

    return np.array(return_list)


def generate_plot(L, xi_range, correlations, name=None):
    if not name:
        name = f"correlation_functions_abs_log_{np.random.randint(0, 10000)}.pdf"

    fig, ax = plt.subplots(figsize=(10, 6))

    L_plot = np.linspace(L // 4 + 1, L, len(correlations[0]))

    ax.set_yscale('log')
    # ax.set_xscale('log')

    for i, g in enumerate(xi_range):
        ax.plot(L_plot, np.abs(correlations[i]), label=f'$\\xi$={g}', color=f'C{i}')
        # ax.plot(L_plot, expected_correlation_function(L_plot - L_for_correlations // 4, g), linestyle='--' ,color=f'C{i}', alpha=0.5)

    # ax.plot(L_plot, np.abs(correlations_NN.real), label='only NN', linestyle=':', color='black')

    ax.set_xlabel("Lattice site $j$")
    ax.set_ylabel("Correlation $|\\langle S^z_{L/4} S^z_j \\rangle|$")

    ax.set_ylim(1e-6, 1.5)

    ax.legend()

    fig.savefig(f"plots/{name}", bbox_inches='tight')


if __name__ == "__main__":
    J = 1
    xi_range = [0.1, 1., 2.0, 5.0, 10., 20.0]
    L_for_correlations_2 = 64

    ground_states_64 = []
    sigmax = np.array([[0., 1.], [1., 0.]])

    for xi in xi_range:
        print(f"Running DMRG simulation for xi = {xi}")
        model, mps, engine = run_dmrg_simulation(L_for_correlations_2, J, xi, tol=1e-10, chi_max=500)
        ground_states_64.append(mps)

    np.save("data/ground_states_64.npy", ground_states_64)

    correlations_64 = [calculate_two_operator_expectation(mps, Sz, Sz, L_for_correlations_2 // 4) for mps in ground_states_64]

    generate_plot(L_for_correlations_2, xi_range, correlations_64)
