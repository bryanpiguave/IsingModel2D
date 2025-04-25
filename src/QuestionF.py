import numpy as np
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
import matplotlib.pyplot as plt
from plot_aesthetics import axis_fontdict, title_fontdict

@njit
def metropolis_step(lattice, J, H, beta, rng_state):
    L = lattice.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        spin = lattice[i, j]
        # periodic neighbors
        neighbors_sum = (lattice[(i + 1) % L, j] + lattice[(i - 1) % L, j]
                         + lattice[i, (j + 1) % L] + lattice[i, (j - 1) % L])
        delta_E = 2 * spin * (J * neighbors_sum + H)
        if delta_E < 0 or np.random.random() < np.exp(-beta * delta_E):
            lattice[i, j] = -spin
    return lattice


def run_single(params):
    T, r, L, J, H, steps, equil_steps, seed = params
    beta = 1.0 / T
    rng = np.random.default_rng(seed)
    # initialize random lattice
    lattice = rng.choice([-1, 1], size=(L, L))
    # equilibration
    for _ in range(equil_steps):
        lattice = metropolis_step(lattice, J, H, beta, None)
    # sampling
    corr_list = []
    for _ in range(steps):
        lattice = metropolis_step(lattice, J, H, beta, None)
        corr = 0.0
        count = 0
        for i in range(L):
            for j in range(L):
                if i + r < L:
                    corr += lattice[i, j] * lattice[i + r, j]
                    count += 1
                if j + r < L:
                    corr += lattice[i, j] * lattice[i, j + r]
                    count += 1
        if count > 0:
            corr_list.append(corr / count)
    if corr_list:
        return T, r, np.mean(corr_list)
    else:
        return T, r, np.nan

if __name__ == "__main__":
    # parameters
    L = 20  # Increased lattice size
    J = 1.0
    H = 0.0
    seed_base = 50
    num_samples = 50  # Increased sampling steps
    equilibration_steps = 50000  # Increased equilibration steps
    num_temps = 7
    temperatures = np.linspace(1.0, 3.5, num_temps)
    distances = np.arange(1, L // 2) # Adjusted distance range

    num_runs = 5 # Number of independent runs per (T, r)

    # prepare parameter list for parallel execution
    param_list = []
    for idx, T in enumerate(temperatures):
        for r in distances:
            for run in range(num_runs):
                param_list.append((T, r, L, J, H, num_samples, equilibration_steps, seed_base + idx + run * num_temps * len(distances)))

    # run in parallel using all available cores
    results = Parallel(n_jobs=-1, verbose=10)(delayed(run_single)(p) for p in param_list)

    # collect into DataFrame
    df = pd.DataFrame(results, columns=['T', 'distance', 'correlation'])

    # Average over independent runs
    averaged_df = df.groupby(['T', 'distance'])['correlation'].mean().reset_index()
    averaged_df.to_csv('outputs/averaged_correlation_function_results.csv', index=False)
    print("Finished simulations. Averaged results saved to outputs/averaged_correlation_function_results.csv")

    # Plotting
    plt.figure(figsize=(10, 6))
    for T in temperatures:
        subset = averaged_df[averaged_df['T'] == T]
        plt.plot(subset['distance'], subset['correlation'], marker='o', linestyle='-', linewidth=1, markersize=4, label=f'T={T:.2f}')
    plt.xlabel('Distance', fontdict=axis_fontdict)
    plt.ylabel('Correlation', fontdict=axis_fontdict)
    plt.title('Averaged Correlation Function vs Distance', fontdict=title_fontdict)
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/averaged_correlation_function_plot.png')
