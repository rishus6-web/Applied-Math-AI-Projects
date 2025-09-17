"""
pinn_heat_equation_polished.py

An implementation of a Physics-Informed Neural Network (PINN)
to solve the 2D Heat Equation.

Author: Rishu Saxena
Date: 2025-09-17
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Configuration Block ---
# All hyperparameters and settings are grouped here for easy modification.
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10000
    LEARNING_RATE = 1e-3
    LR_SCHEDULER_STEP = 5000
    LR_SCHEDULER_GAMMA = 0.1
    ALPHA = 0.1  # Thermal diffusivity
    LAMBDA_PHYSICS = 0.01
    
    # Domain
    T_DOMAIN = [0.0, 1.0]
    X_DOMAIN = [-1.0, 1.0]
    Y_DOMAIN = [-1.0, 1.0]
    
    # Data Generation
    N_IC_POINTS = 2000
    N_BC_POINTS = 2000
    N_COLLOCATION_POINTS = 5000
    IC_SQUARE_SIZE = 0.4 # The hot square is from -0.4 to 0.4

    # Visualization
    VIS_GRID_POINTS = 100
    VIS_FRAMES = 100

# --- 2. The PINN Model ---
class PINN(nn.Module):
    """A standard feed-forward neural network for the PINN."""
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- 3. Loss Functions ---
def compute_data_loss(model, ic_points, ic_values, bc_points, bc_values):
    loss_criterion = nn.MSELoss()
    u_pred_ic = model(ic_points)
    loss_ic = loss_criterion(u_pred_ic, ic_values)
    u_pred_bc = model(bc_points)
    loss_bc = loss_criterion(u_pred_bc, bc_values)
    return loss_ic + loss_bc

def compute_physics_loss(model, collocation_points, alpha):
    points = collocation_points.clone().detach().requires_grad_(True)
    u = model(points)
    grads = torch.autograd.grad(outputs=u, inputs=points, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t, u_x, u_y = grads[:, 0], grads[:, 1], grads[:, 2]
    u_xx = torch.autograd.grad(outputs=u_x, inputs=points, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1]
    u_yy = torch.autograd.grad(outputs=u_y, inputs=points, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 2]
    pde_residual = u_t - alpha * (u_xx + u_yy)
    return torch.mean(pde_residual**2)

# --- 4. Helper Functions ---
def generate_training_data(config):
    """
    Generates the initial, boundary, and collocation points for training.
    """
    # Initial Condition
    ic_points_xy = (torch.rand(config.N_IC_POINTS, 2, device=config.DEVICE) * (config.X_DOMAIN[1] - config.X_DOMAIN[0]) + config.X_DOMAIN[0])
    ic_points_t = torch.full((config.N_IC_POINTS, 1), config.T_DOMAIN[0], device=config.DEVICE)
    ic_points = torch.cat([ic_points_t, ic_points_xy], dim=1)
    ic_values = torch.zeros(config.N_IC_POINTS, 1, device=config.DEVICE)
    ic_values[(ic_points[:, 1].abs() < config.IC_SQUARE_SIZE) & (ic_points[:, 2].abs() < config.IC_SQUARE_SIZE)] = 1.0

    # Boundary Condition
    bc_points = torch.rand(config.N_BC_POINTS, 3, device=config.DEVICE)
    bc_points[:, 0] = bc_points[:, 0] * (config.T_DOMAIN[1] - config.T_DOMAIN[0]) + config.T_DOMAIN[0]
    bc_points[:, 1:] = (bc_points[:, 1:] * (config.X_DOMAIN[1] - config.X_DOMAIN[0]) + config.X_DOMAIN[0])
    edge_choice = torch.randint(0, 4, (config.N_BC_POINTS,))
    bc_points[edge_choice == 0, 1] = config.X_DOMAIN[0]
    bc_points[edge_choice == 1, 1] = config.X_DOMAIN[1]
    bc_points[edge_choice == 2, 2] = config.Y_DOMAIN[0]
    bc_points[edge_choice == 3, 2] = config.Y_DOMAIN[1]
    bc_values = torch.zeros(config.N_BC_POINTS, 1, device=config.DEVICE)

    # Collocation Points
    collocation_points_t = torch.rand(config.N_COLLOCATION_POINTS, 1, device=config.DEVICE) * (config.T_DOMAIN[1] - config.T_DOMAIN[0]) + config.T_DOMAIN[0]
    collocation_points_xy = (torch.rand(config.N_COLLOCATION_POINTS, 2, device=config.DEVICE) * (config.X_DOMAIN[1] - config.X_DOMAIN[0]) + config.X_DOMAIN[0])
    collocation_points = torch.cat([collocation_points_t, collocation_points_xy], dim=1)
    
    return ic_points, ic_values, bc_points, bc_values, collocation_points

def train_pinn(model, config, data):
    """
    The main training loop for the PINN model.
    """
    ic_points, ic_values, bc_points, bc_values, collocation_points = data
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=config.LR_SCHEDULER_STEP, gamma=config.LR_SCHEDULER_GAMMA)
    
    epoch_list, loss_history = [], []
    print("Starting training...")
    for epoch in range(config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        loss_data = compute_data_loss(model, ic_points, ic_values, bc_points, bc_values)
        loss_physics = compute_physics_loss(model, collocation_points, config.ALPHA)
        total_loss = loss_data + config.LAMBDA_PHYSICS * loss_physics
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{config.EPOCHS}], LR: {scheduler.get_last_lr()[0]:.1e}, Loss: {total_loss.item():.4f}")
            epoch_list.append(epoch + 1)
            loss_history.append(total_loss.item())
            
    print("Training finished.")
    return epoch_list, loss_history

def plot_time_evolution(model, config):
    """Generates a plot of temperature evolution at specific points."""
    print("Generating time evolution plot...")
    model.eval()
    
    points_to_track = {
        "Center (0.0, 0.0)": [0.0, 0.0],
        "Point (0.5, 0.5)": [0.5, 0.5],
        "Point (-0.5, -0.5)": [-0.5, -0.5]
    }
    
    time_steps = torch.linspace(config.T_DOMAIN[0], config.T_DOMAIN[1], config.VIS_FRAMES).to(config.DEVICE)
    
    plt.figure(figsize=(10, 6))
    
    with torch.no_grad():
        for label, (px, py) in points_to_track.items():
            input_x = torch.full_like(time_steps, px)
            input_y = torch.full_like(time_steps, py)
            input_tensor = torch.stack((time_steps, input_x, input_y), dim=1)
            
            temp_history = model(input_tensor).cpu().numpy()
            plt.plot(time_steps.cpu().numpy(), temp_history, label=label)

    plt.title('PINN: Temperature Evolution at Specific Points')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (u)')
    plt.legend()
    plt.grid(True)
    plt.savefig('PINN_time_evolution.png')
    plt.close()
    print("Time evolution plot saved as 'PINN_time_evolution.png'")

def generate_animation(model, config):
    """Generates an MP4 animation of the heat diffusion."""
    print("Generating animation...")
    model.eval()

    x = torch.linspace(config.X_DOMAIN[0], config.X_DOMAIN[1], config.VIS_GRID_POINTS)
    y = torch.linspace(config.Y_DOMAIN[0], config.Y_DOMAIN[1], config.VIS_GRID_POINTS)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    time_steps = torch.linspace(config.T_DOMAIN[0], config.T_DOMAIN[1], config.VIS_FRAMES)

    all_predictions = []
    with torch.no_grad():
        for t in time_steps:
            t_tensor = torch.full_like(xx.flatten(), fill_value=t)
            input_tensor = torch.stack((t_tensor, xx.flatten(), yy.flatten()), dim=1).to(config.DEVICE)
            u_pred = model(input_tensor)
            all_predictions.append(u_pred.reshape(config.VIS_GRID_POINTS, config.VIS_GRID_POINTS).cpu().numpy())

    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(all_predictions[0], cmap='hot', interpolation='nearest',
                    extent=[config.X_DOMAIN[0], config.X_DOMAIN[1], config.Y_DOMAIN[0], config.Y_DOMAIN[1]],
                    origin='lower', vmin=0, vmax=1)
    fig.colorbar(img, label='Temperature (u)')
    ax.set_title(f'Temperature at t = {time_steps[0]:.2f} s')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def update(frame):
        img.set_array(all_predictions[frame])
        ax.set_title(f'Temperature at t = {time_steps[frame]:.2f} s')
        return [img]

    anim = FuncAnimation(fig, update, frames=config.VIS_FRAMES, blit=True)
    anim.save('PINN_heat_equation.mp4', writer='ffmpeg', fps=15)
    plt.close()
    print("Animation saved as 'PINN_heat_equation.mp4'")

def visualize_results(model, config, loss_data):
    """
    Generates and saves all visualizations for the project.
    """
    print("\n--- Generating Visualizations ---")
    
    # 1. Loss History Plot
    epochs, losses = loss_data
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses)
    plt.yscale('log')
    plt.title('PINN Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss (log scale)')
    plt.grid(True)
    plt.savefig('PINN_loss_history.png')
    plt.close()
    print("Loss history plot saved as 'PINN_loss_history.png'")
    
    # 2. Time Evolution Plot
    plot_time_evolution(model, config)
    
    # 3. Animation Video
    generate_animation(model, config)
    
    print("\nAll visualizations generated.")


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # Create the configuration object
    config = Config()
    print(f"Using device: {config.DEVICE}")

    # Instantiate the model
    pinn_model = PINN().to(config.DEVICE)
    
    # Generate the data
    training_data = generate_training_data(config)
    
    # Train the model
    epoch_history, loss_history = train_pinn(pinn_model, config, training_data)
    
    # Visualize the results
    visualize_results(pinn_model, config, (epoch_history, loss_history))

