"""Implementing the Heisenberg model with
 exponentially decaying interactions for use in DMRG simulations."""

import numpy as np

class HeisenbergModel:
    def __init__(self, L, J, xi):
        self.L = L
        self.d = 2  # Local dimension (spin-1/2)
        self.J = J  # Coupling constant
        self.xi = xi  # Correlation length
        self.Sx = 0.5 * np.array([[0., 1.], [1., 0.]])
        self.Sy = 0.5 * np.array([[0., -1j], [1j, 0.]])
        self.Sz = 0.5 * np.array([[1., 0.], [0., -1.]])
        self.id = np.eye(2)
        self.H_mpo = self.generate_H_mpos()

    def generate_H_mpos(self) -> list[np.ndarray]:
        """Generate the MPO representation of the Heisenberg Hamiltonian."""
        D = 5
        lam = np.exp(-1 / self.xi)

        W = np.zeros((D, D, 2, 2), dtype=np.complex128)

        # Setting the entries of W according to the Heisenberg model
        W[0, 0] = self.id
        W[0, 1] = self.Sx
        W[0, 2] = self.Sy
        W[0, 3] = self.Sz
        W[1, -1] = self.J * self.Sx
        W[2, -1] = self.J * self.Sy
        W[3, -1] = self.J * self.Sz
        W[4, 4] = self.id

        # now set the "A" part of W
        W[1,1] = W[2,2] = W[3,3] = lam * self.id

        # Return a list of W for each bond
        return [W for _ in range(self.L)]
    
    def energy(self, psi):
        """Evaluate energy E = <psi|H|psi> for the given MPS."""
        assert psi.L == self.L
        
        raise NotImplementedError("Energy calculation not implemented yet.")

