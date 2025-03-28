import numpy as np
import scipy.constants as const
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class PoissonSolver2D:
    """Solves the 2D Poisson equation using Gauss-Seidel with custom boundary conditions"""
    def __init__(self, grid_shape, epsilon, doping_profile, bias_boundaries, tol=1e-3, max_iter=1000):
        self.grid_shape = grid_shape
        self.epsilon = epsilon
        self.doping_profile = doping_profile
        self.bias_boundaries = bias_boundaries
        self.tol = tol
        self.max_iter = max_iter
        self.phi = np.zeros(grid_shape)
        self.rho = np.zeros(grid_shape)

    def boundary_gate(self, x, y):
        return self.bias_boundaries[0] if (int(self.grid_shape[0] / 3.0) < x < 66) and (y == 0) else 0

    def boundary_source(self, x, y):
        return self.bias_boundaries[1] if (x < int(self.grid_shape[0] / 3.0)) and (y == self.grid_shape[1] - 1) else 0

    def boundary_drain(self, x, y):
        return self.bias_boundaries[2] if (x > int(2 * self.grid_shape[0] / 3.0)) and (y == self.grid_shape[1] - 1) else 0

    def boundary_top(self, x, y):
        return self.bias_boundaries[3] if (int(self.grid_shape[0] / 3.0) < x < int(2 * self.grid_shape[0] / 3.0)) and (y == self.grid_shape[1] - 1) else 0

    def _set_boundary_conditions(self):
        X, Y = self.grid_shape
        for i in range(X):
            self.phi[0, i] = self.boundary_gate(i, 0)
            self.phi[Y - 1, i] = self.boundary_top(i, Y - 1) + self.boundary_source(i, Y - 1) + self.boundary_drain(i, Y - 1)

    def solve(self, electron_density):
        q = const.e
        self.rho = q * (self.doping_profile - electron_density) / self.epsilon

        iteration = 0
        error = np.inf
        while iteration < self.max_iter and error > self.tol:
            phi_prime = self.phi.copy()
            for i in range(1, self.grid_shape[0] - 1):
                for j in range(1, self.grid_shape[1] - 1):
                    if self.boundary_source(i, j):
                        phi_prime[i, j] = self.bias_boundaries[1]
                    elif self.boundary_drain(i, j):
                        phi_prime[i, j] = self.bias_boundaries[2]
                    elif self.boundary_gate(i, j):
                        phi_prime[i, j] = self.bias_boundaries[0]
                    elif self.boundary_top(i, j):
                        phi_prime[i, j] = self.bias_boundaries[3]
                    else:
                        phi_prime[i, j] = 0.25 * (
                            self.phi[i + 1, j] + self.phi[i - 1, j] +
                            self.phi[i, j + 1] + self.phi[i, j - 1] -
                            self.rho[i, j])
            self._set_boundary_conditions()
            error = np.max(np.abs(phi_prime - self.phi))
            self.phi = phi_prime
            iteration += 1

        return self.phi



class SchrodingerSolver2D:
    """Solves the 2D Schrödinger equation for energy levels and wavefunctions"""
    def __init__(self, grid_shape, effective_mass):
        self.grid_shape = grid_shape
        self.effective_mass = effective_mass

    def solve(self, potential):
        """Solves the 2D Schrödinger equation"""
        Nx, Ny = self.grid_shape
        dx = dy = 1e-9
        N = Nx * Ny

        H = sp.lil_matrix((N, N))

        for i in range(N):
            H[i, i] = 4 / dx ** 2 + potential.flatten()[i]
            if i % Nx != 0:
                H[i, i - 1] = -1 / dx ** 2
            if (i + 1) % Nx != 0:
                H[i, i + 1] = -1 / dx ** 2
            if i >= Nx:
                H[i, i - Nx] = -1 / dx ** 2
            if i < N - Nx:
                H[i, i + Nx] = -1 / dx ** 2

        H *= -const.hbar ** 2 / (2 * self.effective_mass * const.m_e)
        H = sp.csr_matrix(H)

        eigenvalues, eigenvectors = spla.eigsh(H, k=10, which='SM')
        wavefunctions = eigenvectors.reshape((Nx, Ny, -1))

        return eigenvalues, wavefunctions


class ElectronDensity2D:
    """Computes 2D electron density using Fermi-Dirac distribution"""
    def __init__(self, temperature, fermi_level):
        self.temperature = temperature
        self.fermi_level = fermi_level

    def compute(self, eigenvalues, wavefunctions):
        """Compute new electron density"""
        fermi_dist = 1 / (np.exp((eigenvalues - self.fermi_level) / (const.k * self.temperature)) + 1)
        density = np.sum(np.abs(wavefunctions) ** 2 * fermi_dist[np.newaxis, np.newaxis, :], axis=2)
        return density


class SchrodingerPoissonSolver2D:
    """Self-consistent 2D Schrödinger-Poisson solver"""
    def __init__(self, grid_shape, epsilon, effective_mass, doping_profile, temperature, fermi_level, bias_boundaries, tol=1e-5, max_iter=100):
        self.grid_shape = grid_shape
        self.poisson = PoissonSolver2D(grid_shape, epsilon, doping_profile, bias_boundaries)
        self.schrodinger = SchrodingerSolver2D(grid_shape, effective_mass)
        self.electron_density_solver = ElectronDensity2D(temperature, fermi_level)
        self.tol = tol
        self.max_iter = max_iter

    def solve(self):
        """Solves the coupled Schrödinger-Poisson system iteratively"""
        Nx, Ny = self.grid_shape
        n_old = np.zeros((Nx, Ny))

        for iteration in range(self.max_iter):
            phi = self.poisson.solve(n_old)
            energy_levels, wavefunctions = self.schrodinger.solve(phi)
            n_new = self.electron_density_solver.compute(energy_levels, wavefunctions)

            error = np.linalg.norm(n_new - n_old)
            print(f"Iteration {iteration + 1}: Error = {error}")

            if error < self.tol:
                print("Converged!")
                break

            n_old = n_new

        return phi, energy_levels, wavefunctions, n_new