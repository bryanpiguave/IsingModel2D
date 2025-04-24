import numpy as np
from ising_model import Ising2D
from QuestionD import calculate_magnetization_and_susceptibility
import pandas as pd
import matplotlib.pyplot as plt
"""
Use finite size scaling to determine the critical temperature Tc in the limit L → ∞
and the critical exponent β defined by ⟨|ML|⟩ ∼ |T − Tc|β.
Compare with Onsager’s solution.

"""
def main():
    # Define the lattice sizes and temperature range
    Length_vector = np.array([10, 20, 50, 100, 1000])
    Temperature_vector = np.linspace(1.5, 3.5, 41)  # Fine grid around Tc_exact ≈ 2.269
    # Initialize arrays to store results
    avg_abs_mag_per_site = np.zeros((len(Length_vector), len(Temperature_vector)))
    susceptibility_per_site = np.zeros((len(Length_vector), len(Temperature_vector)))
    # Loop over lattice sizes and temperatures
    for i, L in enumerate(Length_vector):
        for j, T in enumerate(Temperature_vector):
            # Calculate the average absolute magnetization and susceptibility
            avg_abs_mag_per_site[i, j], susceptibility_per_site[i, j] = calculate_magnetization_and_susceptibility(L, T, num_steps=50, equilibration_steps=1000)
    # Save the results to a CSV file
    results = {
        'L': Length_vector,
        'T': Temperature_vector,
        'avg_abs_mag_per_site': avg_abs_mag_per_site.flatten(),
        'susceptibility_per_site': susceptibility_per_site.flatten()
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/finite_size_scaling_results.csv', index=False)
    print("Results saved to 'outputs/finite_size_scaling_results.csv'")

    # Plot the results in log log scale
    Tc_exact = 2.269
    abs_diff = np.abs(Temperature_vector - Tc_exact)
    beta = 0.325  # Critical exponent for the 2D Ising model
    plt.figure(figsize=(12, 6))
    for i, L in enumerate(Length_vector):
        plt.plot(abs_diff, avg_abs_mag_per_site[i], label=f'L={L}')
    plt.plot(abs_diff, abs_diff**beta, 'k--', label='β = 0.325 (Onsager)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('|T - Tc|')
    plt.ylabel('⟨|ML|⟩')
    plt.title('Finite Size Scaling of the 2D Ising Model')
    plt.legend()





    return 


if __name__ == "__main__":
    main()