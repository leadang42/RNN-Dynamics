import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os

# Global parameters
tau = 20e-3  # Time constant (20 ms)
m = 200      # Number of neurons per population
n = 200      # Network size
sigma = 1    # Noise standard deviation
kappa = np.pi/4  # Tuning width
alpha = 0.9      # Target spectral radius
alpha_prime = 0.9  # Target spectral radius for balanced network
dt = 1e-3    # Time step (1 ms)
T = 60e-3    # Total simulation time (60 ms)
theta = np.pi  # Stimulus orientation
num_time_steps = int(T/dt) + 1

# Core network functions
def rescale_matrix(W, target_alpha):
    """Rescale matrix W to have spectral abscissa target_alpha."""
    if np.all(W == 0):
        return W
    eigenvalues = linalg.eigvals(W)
    current_alpha = np.max(np.real(eigenvalues))
    return W * (target_alpha / current_alpha) if current_alpha != 0 else W

def orientation_encoding(phi, theta, kappa):
    """Compute the orientation encoding h(θ) using von Mises-like function."""
    return np.exp((np.cos(phi - theta) - 1) / kappa**2)

def create_model(model_num, m, phi, alpha, alpha_prime):
    """Create weight matrix W and input/output matrices B, C for specified model."""
    if model_num == 1:  # No recurrence
        W = np.zeros((m, m))
        B = np.eye(m)
        C = np.eye(m)
        n_model = m
    
    elif model_num == 2:  # Random symmetric
        np.random.seed(42)
        W_tilde = np.random.normal(0, 1, (m, m))
        W = rescale_matrix(W_tilde + W_tilde.T, alpha)
        B = np.eye(m)
        C = np.eye(m)
        n_model = m
    
    elif model_num == 3:  # Symmetric ring
        W_tilde = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                W_tilde[i, j] = orientation_encoding(phi[i], phi[j], kappa)
        W = rescale_matrix(W_tilde, alpha)
        B = np.eye(m)
        C = np.eye(m)
        n_model = m
    
    elif model_num == 4:  # Balanced ring
        n_model = 2*m
        W_tilde = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                W_tilde[i, j] = orientation_encoding(phi[i], phi[j], kappa)
        W_ring = rescale_matrix(W_tilde, alpha_prime)
        
        W = np.zeros((n_model, n_model))
        W[:m, :m] = W_ring
        W[:m, m:] = -W_ring
        W[m:, :m] = W_ring
        W[m:, m:] = -W_ring
        
        B = np.zeros((n_model, m))
        B[:m, :] = np.eye(m)
        C = np.zeros((m, n_model))
        C[:, :m] = np.eye(m)
        
    return W, B, C, n_model

def simulate_model(model_num, phi, h_theta, tau, dt, num_time_steps):
    """Simulate network dynamics for specified model."""
    W, B, C, n_model = create_model(model_num, m, phi, alpha, alpha_prime)
    r = np.zeros((n_model, num_time_steps))
    r[:, 0] = np.dot(B, h_theta)
    
    for i in range(1, num_time_steps):
        dr = (-r[:, i-1] + np.dot(W, r[:, i-1])) * (dt/tau)
        r[:, i] = r[:, i-1] + dr
    
    return r, C, W

# Added functions for Question 4
def decode_orientation(o_tilde, phi):
    """Decode orientation from noisy readout using Equation (4) from the assignment."""
    numerator = np.sum(o_tilde * np.sin(phi))
    denominator = np.sum(o_tilde * np.cos(phi))
    return np.arctan2(numerator, denominator)

def circular_distance(theta_hat, theta):
    """Compute circular distance between true and estimated orientation using Equation (4)."""
    return np.arccos(np.cos(theta_hat - theta))

def compute_decoding_error_over_time(model_num, phi, h_theta, theta, tau, dt, num_time_steps, sigma, num_trials=100):
    """Compute decoding error over time for a given model, averaged over multiple trials."""
    W, B, C, n_model = create_model(model_num, m, phi, alpha, alpha_prime)
    errors = np.zeros((num_trials, num_time_steps))
    
    for trial in range(num_trials):
        # Run network dynamics
        r = np.zeros((n_model, num_time_steps))
        r[:, 0] = np.dot(B, h_theta)
        
        for i in range(1, num_time_steps):
            dr = (-r[:, i-1] + np.dot(W, r[:, i-1])) * (dt/tau)
            r[:, i] = r[:, i-1] + dr
        
        # Add noise and decode for each time step
        for t in range(num_time_steps):
            # Generate noisy readout
            noise = sigma * np.random.randn(m)
            o_tilde = np.dot(C, r[:, t]) + noise
            
            # Decode orientation
            theta_hat = decode_orientation(o_tilde, phi)
            
            # Compute error
            errors[trial, t] = circular_distance(theta_hat, theta)
    
    # Average across trials
    mean_error = np.mean(errors, axis=0)
    return mean_error

def plot_decoding_errors(phi, h_theta, theta, tau, dt, num_time_steps, sigma):
    """Plot the decoding error over time for all four models."""
    # Compute average decoding error for each model
    error_curves = []
    for model_num in range(1, 5):
        error = compute_decoding_error_over_time(model_num, phi, h_theta, theta, tau, dt, num_time_steps, sigma)
        error_curves.append(error)
    
    # Plot results
    time_points = np.arange(num_time_steps) * dt * 1000  # Convert to ms
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    model_names = ['No Recurrence', 'Random Symmetric', 'Symmetric Ring', 'Balanced Ring']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (error, name) in enumerate(zip(error_curves, model_names)):
        ax.plot(time_points, error, label=f'Model {i+1}: {name}', color=colors[i], linewidth=2)
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Decoding Error (rad)', fontsize=12)
    ax.set_title('Decoding Error vs. Time (θ = π)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig

# Plotting functions (original)
def plot_model_responses(responses, output_weights, phi, theta):
    """Plot responses of all models at specified time points."""
    t_indices = [0, int(20e-3/dt), int(60e-3/dt)]
    t_labels = ['0+ ms', '20 ms', '60 ms']
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    model_names = ['No Recurrence', 'Random Symmetric', 'Symmetric Ring', 'Balanced Ring']
    
    x_ticks = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    x_tick_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    
    for model_idx in range(4):
        for t_idx in range(3):
            ax = axes[model_idx, t_idx]
            output = np.dot(output_weights[model_idx], responses[model_idx][:, t_indices[t_idx]]) if model_idx == 3 else responses[model_idx][:, t_indices[t_idx]]
            
            ax.plot(phi, output, color=colors[model_idx], linewidth=2)
            if model_idx == 0: ax.set_title(t_labels[t_idx], fontsize=12)
            if t_idx == 0: ax.set_ylabel(f'Model {model_idx+1}\n{model_names[model_idx]}', fontsize=11)
            
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels if model_idx == 3 else [])
            ax.set_xlim(0, 2*np.pi)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=theta, color='r', linestyle='--', alpha=0.3)
    
    fig.text(0.5, 0.02, r'Preferred Orientation $\phi$ (rad)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, r'Neural Activity $r(\phi)$', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.1)
    return fig

def plot_comparison_at_60ms(responses, output_weights, phi, theta):
    """Plot comparison of all models at t=60ms."""
    t_idx = int(60e-3/dt)
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = ['No Recurrence', 'Random Symmetric', 'Symmetric Ring', 'Balanced Ring']
    
    x_ticks = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    x_tick_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    
    for model_idx, model_name in enumerate(model_names):
        output = np.dot(output_weights[model_idx], responses[model_idx][:, t_idx]) if model_idx == 3 else responses[model_idx][:, t_idx]
        ax.plot(phi, output, label=f'Model {model_idx+1}: {model_name}', color=colors[model_idx], linewidth=2)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel(r'Preferred Orientation $\phi$ (rad)', fontsize=12)
    ax.set_ylabel(r'Neural Activity $r(\phi)$', fontsize=12)
    ax.set_title('Comparison of Model Responses at t=60ms', fontsize=14)
    ax.legend(loc='best')
    ax.set_xlim(0, 2*np.pi)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=theta, color='r', linestyle='--', alpha=0.5, label=r'Stimulus orientation $\theta=\pi$')
    
    plt.tight_layout()
    return fig

# Modified main execution function to include Question 4
if __name__ == "__main__":
    # Create orientation grid and input
    phi = np.linspace(0, 2*np.pi, m, endpoint=False)
    h_theta = orientation_encoding(phi, theta, kappa)
    
    # Simulate all models
    responses = []
    output_weights = []
    weight_matrices = []
    
    for model_num in range(1, 5):
        r, C, W = simulate_model(model_num, phi, h_theta, tau, dt, num_time_steps)
        responses.append(r)
        output_weights.append(C)
        weight_matrices.append(W)
    
    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate and save plots from the original code
    fig1 = plot_model_responses(responses, output_weights, phi, theta)
    fig2 = plot_comparison_at_60ms(responses, output_weights, phi, theta)
    
    fig1.savefig(os.path.join(results_dir, 'model_responses_grid.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(results_dir, 'model_comparison_60ms.png'), dpi=300, bbox_inches='tight')
    
    # ANSWER TO QUESTION 4: Compute and plot decoding errors
    fig3 = plot_decoding_errors(phi, h_theta, theta, tau, dt, num_time_steps, sigma)
    fig3.savefig(os.path.join(results_dir, 'decoding_error_comparison.png'), dpi=300, bbox_inches='tight')

    plt.show()