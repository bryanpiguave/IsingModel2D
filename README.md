# 2D Ising Model Simulation

This project contains a Python implementation of the 2D Ising model on a square lattice, simulated using the Monte Carlo method with the Metropolis algorithm. It allows for the study of thermodynamic properties such as magnetization and susceptibility as a function of temperature and system size.

## Features

* Simulation of the 2D Ising model on an L x L lattice.
* Periodic boundary conditions.
* Options for different initial spin configurations (random, all up, all down).
* Calculation of energy, magnetization, and susceptibility.
* Monte Carlo simulation using the Metropolis algorithm.
* Functionality to obtain the trajectory of the average spin.
* Analysis of magnetization and susceptibility as a function of temperature for different system sizes.
* Basic estimation of the critical temperature (Tc) and critical exponent (gamma) using finite size scaling (though more detailed analysis would require further implementation).
* (Placeholder for) Calculation of the spin-spin correlation function.

## Getting Started

1.  **Prerequisites:**
    * Python 3.x
    * NumPy (`pip install numpy`)
    * Matplotlib (`pip install matplotlib`)

2.  **Installation:**
    * Clone the repository (if you have one, otherwise just save the provided Python code as a `.py` file, e.g., `ising_model.py`).
    ```bash
    git clone [repository_url]
    cd [repository_directory]
    ```

3.  **Usage:**
    * Run the Python script. The script includes example usage of the implemented functions. You can modify the parameters (lattice size, temperature range, number of steps, etc.) within the script to perform different simulations and analyses.

    ```bash
    python ising_model.py
    ```

    * The script will generate plots of:
        * A histogram to demonstrate the uniformity of the random number generator.
        * The trajectory of the average spin over Monte Carlo steps.
        * Magnetization per site vs. Temperature for different lattice sizes.
        * Susceptibility vs. Temperature for different lattice sizes.

    * The script will also print basic estimates for the critical temperature (Tc) and the critical exponent (gamma).

## Code Structure

* `ising_model.py`: Contains the `Ising2D` class, which implements the Ising model and the Monte Carlo simulation, along with functions to perform the analyses requested in the original problem.

    * `Ising2D(L, initial_state='random', J=1.0, H=0.0)`: Initializes the Ising model.
    * `get_neighbors_sum(i, j)`: Calculates the sum of the nearest neighbor spins.
    * `calculate_energy()`: Calculates the total energy of the lattice.
    * `calculate_magnetization()`: Calculates the total magnetization.
    * `monte_carlo_step(beta)`: Performs one Monte Carlo step using the Metropolis algorithm.
    * `run_simulation(T, num_steps, equilibration_steps=500)`: Runs the simulation and returns the average spin trajectory.
    * `get_lattice()`: Returns the current state of the lattice.
    * `test_random_uniform(num_samples=10000)`: Tests the uniformity of the random number generator.
    * `get_average_spin_trajectory(L=20, T=2.0, num_steps=1000, equilibration_steps=500)`: Simulates and returns the average spin trajectory.
    * `plot_average_spin_trajectory(trajectory, T)`: Plots the average spin trajectory.
    * `calculate_magnetization_and_susceptibility(L, T, num_steps=5000, equilibration_steps=1000)`: Calculates average magnetization and susceptibility.
    * `plot_magnetization_and_susceptibility_vs_T(L_values, T_range)`: Plots magnetization and susceptibility vs. temperature for different L.
    * `estimate_critical_temperature_and_gamma(L_values, T_range, num_steps=5000, equilibration_steps=1000)`: Estimates Tc and gamma.
    * `(Placeholder) calculate_correlation_function(model, r)`: Function to calculate the spin-spin correlation function (currently incomplete).

## Further Work

* Implement the calculation of the spin-spin correlation function as described in Chandler 6.10.
* Perform a more rigorous finite size scaling analysis to accurately determine the critical temperature (Tc) and critical exponents (gamma, nu, etc.). This would involve fitting data to scaling forms.
* Explore different boundary conditions (e.g., free boundary conditions).
* Implement visualization of the lattice configurations during the simulation.
* Optimize the code for performance, especially for larger lattice sizes and longer simulations.

## Acknowledgements

This project was based on the problem statement provided and draws upon concepts from statistical mechanics and computational physics. The reference to Chandler's textbook (likely "Introduction to Modern Statistical Mechanics" by David Chandler) is acknowledged for the context of the correlation function.

## License

[Your License Here (e.g., MIT License)](LICENSE)
