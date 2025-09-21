"""
fdm_heat_equation_polished.py

A professional, well-structured implementation of a classical Finite Difference Method (FDM)
solver for the 2D Heat Equation. This script is designed for clarity, reusability,
and easy comparison with the PINN solver.

Author: Rishu Saxena
Date: 2025-09-19
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Configuration Block ---
# All hyperparameters and settings are grouped here for easy modification.
class Config:
    # Domain parameters
    LX, LY = 2.0, 2.0  # Length of the plate
    NX, NY = 51, 51   # Number of grid points

    # Time parameters
    T_FINAL = 1.0     # Total simulation time
    NT = 2500         # Number of time steps

    # Physical parameters
    ALPHA = 0.1       # Thermal diffusivity

    # Initial Condition
    IC_SQUARE_SIZE = 0.4 # The hot square is from -0.4 to 0.4

    # Visualization
    ANIMATION_FRAME_STEP = 25 # Store a frame every 25 time steps

# --- 2. Helper Functions ---

def initialize_grid(config):
    """
    Initializes the grid and sets the initial condition.

    Args:
        config (Config): An object containing all hyperparameters.

    Returns:
        tuple: A tuple containing the temperature field u, and the x and y coordinate arrays.
    """
    print("Initializing grid and setting initial conditions...")
    # Create the grid
    x = np.linspace(-config.LX / 2, config.LX / 2, config.NX)
    y = np.linspace(-config.LY / 2, config.LY / 2, config.NY)
    
    # Initialize the temperature field `u`
    u = np.zeros((config.NX, config.NY))

    # Set the Initial Condition (IC)
    x_start, x_end = -config.IC_SQUARE_SIZE, config.IC_SQUARE_SIZE
    y_start, y_end = -config.IC_SQUARE_SIZE, config.IC_SQUARE_SIZE
    
    # Find the corresponding grid indices
    ix_start = np.abs(x - x_start).argmin()
    ix_end = np.abs(x - x_end).argmin()
    iy_start = np.abs(y - y_start).argmin()
    iy_end = np.abs(y - y_end).argmin()
    
    u[ix_start:ix_end + 1, iy_start:iy_end + 1] = 1.0
    
    return u, x, y

def run_simulation(config, u, x, y):
    """
    Runs the main time-stepping loop for the FDM simulation.

    Args:
        config (Config): An object containing all hyperparameters.
        u (np.ndarray): The initial temperature field.
        x (np.ndarray): The x-coordinate array.
        y (np.ndarray): The y-coordinate array.

    Returns:
        tuple: Lists containing the history of the temperature field, time steps,
               and total heat.
    """
    dx = config.LX / (config.NX - 1)
    dy = config.LY / (config.NY - 1)
    dt = config.T_FINAL / config.NT

    # Stability Check
    cfl_limit = (dx**2 * dy**2) / (2 * config.ALPHA * (dx**2 + dy**2))
    if dt >= cfl_limit:
        print(f"Warning: Time step is too large. dt = {dt:.6f}")
        print(f"For stability, dt should be less than {cfl_limit:.6f}. The simulation might blow up.")
    else:
        print(f"Stability condition satisfied. dt = {dt:.6f} < {cfl_limit:.6f}")

    u_history = [u.copy()]
    time_history = [0.0]
    heat_history = [np.sum(u)] # Track total heat at t=0

    print("Starting simulation...")
    for n in range(config.NT):
        u_n = u.copy()
        for i in range(1, config.NX - 1):
            for j in range(1, config.NY - 1):
                u_xx = (u_n[i+1, j] - 2*u_n[i, j] + u_n[i-1, j]) / dx**2
                u_yy = (u_n[i, j+1] - 2*u_n[i, j] + u_n[i, j-1]) / dy**2
                u[i, j] = u_n[i, j] + dt * config.ALPHA * (u_xx + u_yy)

        if (n + 1) % config.ANIMATION_FRAME_STEP == 0:
            u_history.append(u.copy())
            time_history.append((n + 1) * dt)
            heat_history.append(np.sum(u)) # Track total heat at this frame
    
    print("Simulation finished.")
    return u_history, time_history, heat_history

def visualize_results(config, u_history, time_history, heat_history, x, y):
    """
    Generates and saves all visualizations for the FDM simulation.
    """
    print("\n--- Generating Visualizations ---")

    # 1. Total Heat (Energy Conservation) Plot
    print("Generating total heat plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(time_history, heat_history)
    plt.title('FDM: Total Heat vs. Time (Energy Conservation Check)')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Heat (Sum of u over grid)')
    plt.grid(True)
    plt.savefig('FDM_total_heat.png')
    plt.close()
    print("Total heat plot saved as 'FDM_total_heat.png'")


    # 2. Time Evolution Comparison Plot
    print("Generating time evolution plot...")
    center_idx = (np.abs(x - 0.0).argmin(), np.abs(y - 0.0).argmin())
    point1_idx = (np.abs(x - 0.5).argmin(), np.abs(y - 0.5).argmin())
    point2_idx = (np.abs(x + 0.5).argmin(), np.abs(y + 0.5).argmin())

    temp_center = [u_frame[center_idx] for u_frame in u_history]
    temp_point1 = [u_frame[point1_idx] for u_frame in u_history]
    temp_point2 = [u_frame[point2_idx] for u_frame in u_history]

    plt.figure(figsize=(10, 6))
    plt.plot(time_history, temp_center, label='Center (0.0, 0.0)')
    plt.plot(time_history, temp_point1, label='Point (0.5, 0.5)')
    plt.plot(time_history, temp_point2, label='Point (-0.5, -0.5)')
    plt.title('FDM: Temperature Evolution at Specific Points')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (u)')
    plt.legend()
    plt.grid(True)
    plt.savefig('FDM_time_evolution.png')
    plt.close()
    print("Time evolution plot saved as 'FDM_time_evolution.png'")

    # 3. Animation Video
    print("Generating animation...")
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(u_history[0].T, cmap='hot', interpolation='nearest',
                    extent=[-config.LX/2, config.LX/2, -config.LY/2, config.LY/2],
                    origin='lower', vmin=0, vmax=1)
    fig.colorbar(img, label='Temperature (u)')
    ax.set_title(f'Temperature at t = {time_history[0]:.2f} s')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def update(frame):
        img.set_array(u_history[frame].T)
        ax.set_title(f'Temperature at t = {time_history[frame]:.2f} s')
        return [img]

    anim = FuncAnimation(fig, update, frames=len(u_history), blit=True)
    anim.save('FDM_heat_equation.mp4', writer='ffmpeg', fps=15)
    plt.close()
    print("Animation saved as 'FDM_heat_equation.mp4'")
    print("\nAll visualizations generated.")

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # Create the configuration object
    config = Config()
    
    # Initialize the grid and IC
    u_initial, x_coords, y_coords = initialize_grid(config)
    
    # Run the simulation
    u_frames, t_frames, heat_frames = run_simulation(config, u_initial, x_coords, y_coords)
    
    # Visualize the results
    visualize_results(config, u_frames, t_frames, heat_frames, x_coords, y_coords)

