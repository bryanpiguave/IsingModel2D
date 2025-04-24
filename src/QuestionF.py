import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Ising2DMetropolis:
    def __init__(self, L=20, J=1.0, H=0.0, initial_state='random', seed=None):
        """
        Initializes the 2D Ising model.

        Args:
            L (int): Linear dimension of the square lattice.
            J (float): Interaction strength between nearest neighbors.
            H (float): External magnetic field strength.
            initial_state (str): 'random' or 'up' or 'down'.
            seed (int, optional): Random seed for reproducibility.
        """
        self.L = L
        self.N = L * L  # Total number of spins
        self.J = J
        self.H = H
        self.beta = 1.0  # Inverse temperature (will be updated)
        self.rng = np.random.default_rng(seed)

        if initial_state == 'random':
            self.lattice = self.rng.choice([-1, 1], size=(L, L))
        elif initial_state == 'up':
            self.lattice = np.ones((L, L), dtype=int)
        elif initial_state == 'down':
            self.lattice = -np.ones((L, L), dtype=int)
        else:
            raise ValueError("Invalid initial_state")

    def _calculate_energy(self):
        """Calculates the total energy of the lattice."""
        energy = 0
        for i in range(self.L):
            for j in range(self.L):
                spin = self.lattice[i, j]
                neighbors_sum = (
                    self.lattice[(i + 1) % self.L, j] +
                    self.lattice[(i - 1) % self.L, j] +
                    self.lattice[i, (j + 1) % self.L] +
                    self.lattice[i, (j - 1) % self.L]
                )
                energy += -self.J * spin * neighbors_sum - self.H * spin
        return energy / 2.0  # Divide by 2 because each bond is counted twice

    def _monte_carlo_step(self):
        """Performs one Monte Carlo step using the Metropolis algorithm."""
        for _ in range(self.N):  # Attempt to flip each spin once on average
            i = self.rng.integers(0, self.L)
            j = self.rng.integers(0, self.L)
            spin = self.lattice[i, j]
            neighbors_sum = (
                self.lattice[(i + 1) % self.L, j] +
                self.lattice[(i - 1) % self.L, j] +
                self.lattice[i, (j + 1) % self.L] +
                self.lattice[i, (j - 1) % self.L]
            )
            delta_E = 2 * spin * (self.J * neighbors_sum + self.H)

            if delta_E < 0:
                self.lattice[i, j] *= -1
            elif self.rng.random() < np.exp(-self.beta * delta_E):
                self.lattice[i, j] *= -1

    def run_simulation(self, T, steps, equilibration_steps=1000):
        """
        Runs the Monte Carlo simulation.

        Args:
            T (float): Temperature of the system.
            steps (int): Number of Monte Carlo steps to perform for data collection.
            equilibration_steps (int): Number of initial steps for equilibration.

        Returns:
            list: A list of lattice configurations sampled during the simulation.
        """
        self.beta = 1.0 / T if T > 0 else float('inf')
        lattice_history = []

        # Equilibration
        for _ in tqdm(range(equilibration_steps), desc=f"Equilibrating at T={T:.2f}"):
            self._monte_carlo_step()

        # Data collection
        for _ in tqdm(range(steps), desc=f"Sampling at T={T:.2f}"):
            self._monte_carlo_step()
            lattice_history.append(self.lattice.copy())

        return lattice_history

    def calculate_average_magnetization(self, lattice_samples):
        """Calculates the average magnetization from the sampled lattices."""
        if not lattice_samples:
            return 0.0
        magnetization_sum = np.sum([np.sum(sample) for sample in lattice_samples])
        return magnetization_sum / (len(lattice_samples) * self.N)

    def calculate_correlation_function(self, lattice_samples, max_distance=None):
        """
        Calculates the correlation function for various spin separations.

        Args:
            lattice_samples (list): List of sampled lattice configurations.
            max_distance (int, optional): Maximum distance to calculate the correlation for.
                                           If None, it calculates for all possible distances.

        Returns:
            dict: A dictionary where keys are the Manhattan distances and values are the
                  corresponding correlation function values.
        """
        if not lattice_samples:
            return {}

        if max_distance is None:
            max_distance = (self.L // 2)  # Maximum Manhattan distance

        correlation_sum = {}
        pair_counts = {}

        for r in range(max_distance + 1):
            correlation_sum[r] = 0.0
            pair_counts[r] = 0

        num_samples = len(lattice_samples)
        avg_magnetization = self.calculate_average_magnetization(lattice_samples)

        for sample in lattice_samples:
            for i in range(self.L):
                for j in range(self.L):
                    s_i = sample[i, j]
                    for k in range(self.L):
                        for l in range(self.L):
                            if (i, j) == (k, l):
                                continue
                            dx = min(abs(i - k), self.L - abs(i - k))
                            dy = min(abs(j - l), self.L - abs(j - l))
                            distance = dx + dy
                            if distance <= max_distance:
                                correlation_sum[distance] += s_i * sample[k, l]
                                pair_counts[distance] += 1

        correlation_function = {}
        for r in range(max_distance + 1):
            if pair_counts[r] > 0:
                correlation_function[r] = (correlation_sum[r] / pair_counts[r]) - (avg_magnetization ** 2)
            else:
                correlation_function[r] = 0.0

        return correlation_function

if __name__ == "__main__":
    L = 20
    model = Ising2DMetropolis(L=L, J=1.0, H=0.0, seed=42)
    steps = 500
    equilibration_steps = 20000
    num_temps = 10
    temperatures = np.linspace(1.5, 3.5, num_temps)  # Range around the critical temperature (Tc ≈ 2.269 for 2D Ising)

    all_magnetizations = {}
    all_correlations = {}

    for T in temperatures:
        print(f"\nRunning simulation for T = {T:.2f}")
        lattice_history = model.run_simulation(T, steps, equilibration_steps)
        avg_mag = model.calculate_average_magnetization(lattice_history)
        correlation = model.calculate_correlation_function(lattice_history, max_distance=L)

        all_magnetizations[T] = avg_mag
        all_correlations[T] = correlation

    # Plotting the results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    temps = list(all_magnetizations.keys())
    mags = list(all_magnetizations.values())
    plt.plot(temps, np.abs(mags), marker='o')
    plt.xlabel("Temperature (T)")
    plt.ylabel("|Average Magnetization (<s>)")
    plt.title("Average Magnetization vs. Temperature")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    distances = range(L + 1)
    for T, corr in all_correlations.items():
        corr_values = [corr.get(d, 0) for d in distances]
        plt.plot(distances, corr_values, marker='.', linestyle='-', label=f"T = {T:.2f}")
    plt.xlabel("Manhattan Distance (r)")
    plt.ylabel("Correlation Function (<s_i s_j> - <s_i><s_j>)")
    plt.title("Correlation Function vs. Distance")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("outputs/answer_questionf.png")

    print("\nAverage Magnetizations:")
    for T, mag in all_magnetizations.items():
        print(f"T = {T:.2f}: <s> = {mag:.4f}")

    print("\nCorrelation Functions (first few distances):")
    for T, corr in all_correlations.items():
        print(f"T = {T:.2f}: {dict(list(corr.items())[:5])}...")