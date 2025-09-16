import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Setup and Parameters ---

# Domain parameters
Lx, Ly = 2.0, 2.0  # Length of the plate in x and y directions
Nx, Ny = 51, 51   # Number of grid points in x and y
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# Time parameters
T = 1.0           # Total simulation time
Nt = 2500         # Number of time steps
dt = T / Nt

# Physical parameters
alpha = 0.1       # Thermal diffusivity

# Stability Check (Courant-Friedrichs-Lewy or CFL condition)
cfl_limit = (dx**2 * dy**2) / (2 * alpha * (dx**2 + dy**2))
if dt >= cfl_limit:
    print(f"Warning: Time step is too large. dt = {dt:.6f}")
    print(f"For stability, dt should be less than {cfl_limit:.6f}. The simulation might blow up.")
else:
    print(f"Stability condition satisfied. dt = {dt:.6f} < {cfl_limit:.6f}")

# --- 2. Initialization ---

# Create the grid
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y)

# Initialize the temperature field `u`
u = np.zeros((Nx, Ny))

# Set the Initial Condition (IC)
# Define the boundaries of the hot square
x_start, x_end = -0.4, 0.4
y_start, y_end = -0.4, 0.4
# Find the corresponding grid indices
ix_start = np.abs(x - x_start).argmin()
ix_end = np.abs(x - x_end).argmin()
iy_start = np.abs(y - y_start).argmin()
iy_end = np.abs(y - y_end).argmin()
# Set the temperature in that region to 1.0
u[ix_start:ix_end+1, iy_start:iy_end+1] = 1.0

# Store history for animation and plotting
u_history = [u.copy()]
time_history = [0.0]

# --- 3. The Time-Stepping Loop ---
print("Starting simulation...")
for n in range(Nt):
    u_n = u.copy()
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u_xx = (u_n[i+1, j] - 2*u_n[i, j] + u_n[i-1, j]) / dx**2
            u_yy = (u_n[i, j+1] - 2*u_n[i, j] + u_n[i, j-1]) / dy**2
            u[i, j] = u_n[i, j] + dt * alpha * (u_xx + u_yy)

    # Store results periodically
    if (n + 1) % 25 == 0:
        u_history.append(u.copy())
        time_history.append((n + 1) * dt)

print("Simulation finished.")

# --- 4. Visualization ---
print("Generating animation...")
fig, ax = plt.subplots(figsize=(6, 5))
img = ax.imshow(u_history[0].T, cmap='hot', interpolation='nearest',
                extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], origin='lower', vmin=0, vmax=1)
fig.colorbar(img, label='Temperature (u)')
ax.set_title(f'Temperature at t = 0.00 s')
ax.set_xlabel('x')
ax.set_ylabel('y')

def update(frame):
    img.set_array(u_history[frame].T)
    ax.set_title(f'Temperature at t = {time_history[frame]:.2f} s')
    return [img]

anim = FuncAnimation(fig, update, frames=len(u_history), blit=True)
# UPDATED FILENAME
anim.save('FDM_heat_equation.mp4', writer='ffmpeg', fps=15)
plt.close()
# UPDATED PRINT STATEMENT
print("\nAnimation saved as 'FDM_heat_equation.mp4'")

# --- 5. Time Evolution Comparison Plot ---
print("Generating time evolution plot...")
# Find grid indices closest to the desired points
center_idx = (np.abs(x - 0.0).argmin(), np.abs(y - 0.0).argmin())
point1_idx = (np.abs(x - 0.5).argmin(), np.abs(y - 0.5).argmin())
point2_idx = (np.abs(x + 0.5).argmin(), np.abs(y + 0.5).argmin())

# Extract temperature history at these points
temp_center = [u[center_idx] for u in u_history]
temp_point1 = [u[point1_idx] for u in u_history]
temp_point2 = [u[point2_idx] for u in u_history]

plt.figure(figsize=(10, 6))
plt.plot(time_history, temp_center, label='Center (0.0, 0.0)')
plt.plot(time_history, temp_point1, label='Point (0.5, 0.5)')
plt.plot(time_history, temp_point2, label='Point (-0.5, -0.5)')
plt.title('FDM: Temperature Evolution at Specific Points')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (u)')
plt.legend()
plt.grid(True)
# UPDATED FILENAME
plt.savefig('FDM_time_evolution.png')
plt.close()
# UPDATED PRINT STATEMENT
print("Time evolution plot saved as 'FDM_time_evolution.png'")



# --- 4. Display the video in Colab (Optional) ---
# This will embed the video directly in your notebook output
try:
    from google.colab import files
    files.download('FDM_time_evolution.png')
except ImportError:
    print("\nTo display the video in a local Jupyter notebook, you might need to run:")
    print("from IPython.display import Video")
    print("Video('FDM_time_evolution.png')")


