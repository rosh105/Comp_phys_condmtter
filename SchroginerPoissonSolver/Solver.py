import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.constants as const

class PoissonSolver2D:
    """Solves the 2D Poisson equation using finite difference method"""
    def __init__(self, grid_shape, epsilon, doping_profile):
        self.grid_shape = grid_shape
        self.epsilon = epsilon
        self.doping_profile = doping_profile
        self.phi = np.zeros(grid_shape)

    def solve(self, electron_density):
        """Solves 2D Poisson equation using finite difference method"""
        Nx, Ny = self.grid_shape
        dx = dy = 1e-9  # 1 nm grid spacing

        N = Nx * Ny
        A = sp.lil_matrix((N, N))

        for i in range(N):
            A[i, i] = -4
            if i % Nx != 0:
                A[i, i - 1] = 1
            if (i + 1) % Nx != 0:
                A[i, i + 1] = 1
            if i >= Nx:
                A[i, i - Nx] = 1
            if i < N - Nx:
                A[i, i + Nx] = 1

        A /= dx ** 2
        A = sp.csr_matrix(A)

        rho = const.e * (self.doping_profile - electron_density) / self.epsilon
        rho_vector = rho.flatten()

        phi_vector = spla.spsolve(A, -rho_vector)
        self.phi = phi_vector.reshape((Nx, Ny))
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
    def __init__(self, grid_shape, epsilon, effective_mass, doping_profile, temperature, fermi_level, tol=1e-5, max_iter=100):
        self.grid_shape = grid_shape
        self.poisson = PoissonSolver2D(grid_shape, epsilon, doping_profile)
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

