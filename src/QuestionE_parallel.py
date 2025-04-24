import numpy as np
from multiprocessing import Pool, cpu_count
from ising_model import Ising2D
from QuestionD import calculate_magnetization_and_susceptibility
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from functools import partial

def compute_for_temperature(L, T, num_steps=50, equilibration_steps=1000):
    """Helper function for parallel computation of a single temperature point"""
    try:
        mag, sus = calculate_magnetization_and_susceptibility(L, T, num_steps, equilibration_steps)
        return {'L': L, 'T': T, 'avg_abs_mag_per_site': mag, 'susceptibility_per_site': sus}
    except Exception as e:
        print(f"Error computing L={L}, T={T}: {str(e)}")
        return None

def main():
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    output_file = 'outputs/parallel_finite_size_scaling_results.csv'
    
    # Define the lattice sizes and temperature range
    Length_vector = np.array([10, 20, 50, 100, 1000])
    Temperature_vector = np.linspace(1.5, 3.5, 41)  # Fine grid around Tc_exact ≈ 2.269
    Length_vector = np.array([ 50, 100])
    Temperature_vector = np.linspace(1.5, 3.5, 5)  # Fine grid around Tc_exact ≈ 2.269
    # Check if results already exist
    if os.path.exists(output_file):
        print("Loading existing results...")
        results_df = pd.read_csv(output_file)
        
        # Check if all required data is present
        required_L = set(Length_vector)
        required_T = set(Temperature_vector)
        existing_L = set(results_df['L'].unique())
        existing_T = set(results_df['T'].unique())
        
        if required_L.issubset(existing_L) and required_T.issubset(existing_T):
            print("All required data found in existing file.")
        else:
            print("Some data missing - will recompute missing points")
            results_df = None
    else:
        results_df = None
    
    # If we need to compute (either missing data or no file)
    if results_df is None:
        print("Starting parallel computation...")
        start_time = time.time()
        
        # Prepare all parameter combinations
        params = []
        for L in Length_vector:
            for T in Temperature_vector:
                # Skip if we already have this data point
                if results_df is not None:
                    existing = results_df[(results_df['L'] == L) & (results_df['T'] == T)]
                    if len(existing) > 0:
                        continue
                params.append((L, T))
        
        # Create partial function with fixed parameters
        compute_func = partial(compute_for_temperature, 
                             num_steps=50, 
                             equilibration_steps=10000)
        
        # Use multiprocessing Pool
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(compute_func, params)
        
        # Filter out None results (failed computations)
        valid_results = [r for r in results if r is not None]
        
        # Combine with existing results if any
        if results_df is not None:
            new_df = pd.DataFrame(valid_results)
            results_df = pd.concat([results_df, new_df], ignore_index=True)
        else:
            results_df = pd.DataFrame(valid_results)
        
        # Save results
        results_df.to_csv(output_file, index=False)
        print(f"Computation completed in {time.time()-start_time:.2f} seconds")
        print(f"Results saved to {output_file}")
    
    # Prepare data for plotting
    # Pivot the data for easier plotting
    mag_data = results_df.pivot(index='T', columns='L', values='avg_abs_mag_per_site')
    sus_data = results_df.pivot(index='T', columns='L', values='susceptibility_per_site')
    
    # Plot magnetization results
    Tc_exact = 2.269
    abs_diff = np.abs(mag_data.index.values - Tc_exact)
    beta = 0.325  # Critical exponent for the 2D Ising model
    
    plt.figure(figsize=(12, 6))
    for L in mag_data.columns:
        plt.plot(abs_diff, mag_data[L], 'o-', label=f'L={L}')
    
    plt.plot(abs_diff, abs_diff**beta, 'k--', label='β = 0.325 (Onsager)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('|T - Tc|')
    plt.ylabel('⟨|ML|⟩')
    plt.title('Finite Size Scaling of the 2D Ising Model')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig('outputs/finite_size_scaling_plot.png')
    plt.close()
    
    # Additional plot: susceptibility vs temperature
    plt.figure(figsize=(12, 6))
    for L in sus_data.columns:
        plt.plot(sus_data.index, sus_data[L], 'o-', label=f'L={L}')
    
    plt.axvline(Tc_exact, color='k', linestyle='--', label='Tc (Onsager)')
    plt.xlabel('Temperature')
    plt.ylabel('Susceptibility per site')
    plt.title('Susceptibility vs Temperature for Different Lattice Sizes')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/susceptibility_vs_temperature.png')
    plt.close()

if __name__ == "__main__":
    main()