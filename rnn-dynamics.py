import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os


### GLOBAL PARAMETERS ###

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

# Generate random weights for model 2 once and for all
np.random.seed(42)
W_tilde_model2 = np.random.normal(0, 1, (m, m))
W_tilde_model2 = (W_tilde_model2 + W_tilde_model2.T) / 2  # Make symmetric

viridis = plt.cm.viridis(np.linspace(0, 0.85, 4))  # Colormap for plotting


### CORE MODEL ###

# 1 Stimulus encoding
def stimulus_encoding(phi, theta, kappa):
    return np.exp((np.cos(phi - theta) - 1) / kappa**2)

# 2 V1 dynamics
def network_model(model_num, m, phi, alpha, alpha_prime):
    
    def rescale_matrix(W, target_alpha):
        if np.all(W == 0):
            return W
        eigenvalues = linalg.eigvals(W)
        current_alpha = np.max(np.real(eigenvalues))
        return W * (target_alpha / current_alpha) if current_alpha != 0 else W

    if model_num == 1:  # No recurrence
        W = np.zeros((m, m))
        B = np.eye(m)
        C = np.eye(m)
        n_model = m
    
    elif model_num == 2:  # Random symmetric
        W = rescale_matrix(W_tilde_model2, alpha)
        B = np.eye(m)
        C = np.eye(m)
        n_model = m
    
    elif model_num == 3:  # Symmetric ring
        W_tilde = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                W_tilde[i, j] = stimulus_encoding(phi[i], phi[j], kappa)
        W = rescale_matrix(W_tilde, alpha)
        B = np.eye(m)
        C = np.eye(m)
        n_model = m
    
    elif model_num == 4:  # Balanced ring
        n_model = 2*m
        W_tilde = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                W_tilde[i, j] = stimulus_encoding(phi[i], phi[j], kappa)
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

    print(f"Model {model_num}: {W}") 
    
    return W, B, C, n_model

def simulate_model(model_num, phi, h_theta, tau, dt, num_time_steps):
    W, B, C, n_model = network_model(model_num, m, phi, alpha, alpha_prime)
    r = np.zeros((n_model, num_time_steps))
    r[:, 0] = np.dot(B, h_theta)
    
    for i in range(1, num_time_steps):
        dr = (-r[:, i-1] + np.dot(W, r[:, i-1])) * (dt/tau)
        r[:, i] = r[:, i-1] + dr
    
    return r, C, W

# 3 Noisy readout
def noisy_readout(r, t, C, sigma):
    noise = sigma * np.random.randn(m)
    o_tilde = np.dot(C, r[:, t]) + noise
    
    return o_tilde

# 4 Stimulus decoding
def decode_orientation(o_tilde, phi):
    numerator = np.sum(o_tilde * np.sin(phi))
    denominator = np.sum(o_tilde * np.cos(phi))
    return np.arctan2(numerator, denominator)

def circular_distance(theta_hat, theta):
    return np.arccos(np.cos(theta_hat - theta))


### ANALYSIS ###

def compute_decoding_error_over_time(model_num, phi, h_theta, theta, tau, dt, num_time_steps, sigma, num_trials=100):
    W, B, C, n_model = network_model(model_num, m, phi, alpha, alpha_prime)
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

            o_tilde = noisy_readout(r, t, C, sigma)
            theta_hat = decode_orientation(o_tilde, phi)
            errors[trial, t] = circular_distance(theta_hat, theta)
    
    # Average across trials
    mean_error = np.mean(errors, axis=0)
    return mean_error


### PLOTTING ###

def plot_model_responses(responses, output_weights, phi, theta, save_both=False):
    t_indices = [0, int(20e-3/dt), int(60e-3/dt)]
    t_labels = ['0+ ms', '20 ms', '60 ms']
    fig, axes = plt.subplots(4, 3, figsize=(6, 7))
    model_names = ['No Recurrence', 'Random Symmetric', 'Symmetric Ring', 'Balanced Ring']
    
    x_ticks = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    x_tick_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    
    # First plot: without background responses
    for model_idx in range(4):
        for t_idx in range(3):
            ax = axes[model_idx, t_idx]
            output = np.dot(output_weights[model_idx], responses[model_idx][:, t_indices[t_idx]]) if model_idx == 3 else responses[model_idx][:, t_indices[t_idx]]
            
            ax.plot(phi, output, color=viridis[model_idx], linewidth=2)
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
    
    if save_both:
        fig.savefig(os.path.join(results_dir, 'model_responses_grid.png'), dpi=300, bbox_inches='tight')
    
    # Second plot: with background responses
    fig2, axes2 = plt.subplots(4, 3, figsize=(6, 7))
    
    for model_idx in range(4):
        for t_idx in range(3):
            ax = axes2[model_idx, t_idx]
            
            # Plot other models in background
            for other_idx in range(4):
                if other_idx != model_idx:
                    other_output = np.dot(output_weights[other_idx], responses[other_idx][:, t_indices[t_idx]]) if other_idx == 3 else responses[other_idx][:, t_indices[t_idx]]
                    ax.plot(phi, other_output, color='lightgrey', linewidth=1, alpha=0.5)
            
            # Plot current model in foreground
            output = np.dot(output_weights[model_idx], responses[model_idx][:, t_indices[t_idx]]) if model_idx == 3 else responses[model_idx][:, t_indices[t_idx]]
            ax.plot(phi, output, color=viridis[model_idx], linewidth=2)
            
            if model_idx == 0: ax.set_title(t_labels[t_idx], fontsize=12)
            if t_idx == 0: ax.set_ylabel(f'Model {model_idx+1}\n{model_names[model_idx]}', fontsize=11)
            
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels if model_idx == 3 else [])
            ax.set_xlim(0, 2*np.pi)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=theta, color='r', linestyle='--', alpha=0.3)
    
    fig2.text(0.5, 0.02, r'Preferred Orientation $\phi$ (rad)', ha='center', fontsize=12)
    fig2.text(0.02, 0.5, r'Neural Activity $r(\phi)$', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.1)
    
    if save_both:
        fig2.savefig(os.path.join(results_dir, 'model_responses_grid_with_background.png'), dpi=300, bbox_inches='tight')
    
    return fig, fig2

def plot_comparison_at_60ms(responses, output_weights, phi, theta):
    t_idx = int(60e-3/dt)
    fig, ax = plt.subplots(figsize=(6, 10))
    model_names = ['No Recurrence', 'Random Symmetric', 'Symmetric Ring', 'Balanced Ring']
    
    x_ticks = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    x_tick_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    
    for model_idx, model_name in enumerate(model_names):
        output = np.dot(output_weights[model_idx], responses[model_idx][:, t_idx]) if model_idx == 3 else responses[model_idx][:, t_idx]
        ax.plot(phi, output, label=f'Model {model_idx+1}: {model_name}', color=viridis[model_idx], linewidth=2)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel(r'Preferred Orientation $\phi$ (rad)', fontsize=12)
    ax.set_ylabel(r'Neural Activity $r(\phi)$', fontsize=12)
    ax.set_title('Comparison of Model Responses at t=60ms', fontsize=14)
    # ax.legend(loc='best')
    ax.set_xlim(0, 2*np.pi)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=theta, color='r', linestyle='--', alpha=0.5, label=r'Stimulus orientation $\theta=\pi$')
    
    plt.tight_layout()
    return fig

def plot_decoding_errors(phi, h_theta, theta, tau, dt, num_time_steps, sigma):
    # Compute average decoding error for each model
    error_curves = []
    for model_num in range(1, 5):
        error = compute_decoding_error_over_time(model_num, phi, h_theta, theta, tau, dt, num_time_steps, sigma)
        error_curves.append(error)
    
    # Plot results
    time_points = np.arange(num_time_steps) * dt * 1000  # Convert to ms
    model_names = ['No Recurrence', 'Random Symmetric', 'Symmetric Ring', 'Balanced Ring']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for i, (error, name) in enumerate(zip(error_curves, model_names)):
        ax.plot(time_points, error, label=f'Model {i+1}: {name}', color=viridis[i], linewidth=2)
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Decoding Error (rad)', fontsize=12)
    ax.set_title('Decoding Error vs. Time (θ = π)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig

def compute_eigenvalues(model_num, m, phi, alpha, alpha_prime):
    W, _, _, _ = network_model(model_num, m, phi, alpha, alpha_prime)
    eigenvalues = linalg.eigvals(W)

    # print(f"Eigenvalues of Model {model_num}: ", eigenvalues)

    return eigenvalues

def plot_eigenvalues(phi, alpha, alpha_prime):
    model_names = ['No Recurrence', 'Random Symmetric', 'Symmetric Ring', 'Balanced Ring']
    figs = []
    
    for model_num in range(1, 5):
        fig, ax = plt.subplots(figsize=(3,3))
        eigenvalues = compute_eigenvalues(model_num, m, phi, alpha, alpha_prime)
        
        # Calculate actual abscissa limit
        actual_alpha = np.max(np.real(eigenvalues))
        
        # Plot real vs imaginary parts
        ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), 
                  alpha=0.6, s=20, label=f'Model {model_num}\nα = {actual_alpha:.3f}')
        
        # Add unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
        
        ax.set_xlabel('Re(λ)', fontsize=12)
        ax.set_ylabel('Im(λ)', fontsize=12)
        ax.set_title(f'{model_names[model_num-1]}', fontsize=14)
        ax.grid(True, alpha=0.3)
        # ax.legend()
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Set axis limits to show full unit circle
        ax.set_xlim(-5.2, 5.2)
        ax.set_ylim(-5.2, 5.2)
        
        plt.tight_layout()
        figs.append(fig)
    
    return figs

def plot_eigenvectors(phi, alpha, alpha_prime):
    model_names = ['No Recurrence', 'Random Symmetric', 'Symmetric Ring', 'Balanced Ring']
    figs = []
    
    # Define colors from viridis colormap
    real_color = viridis[1]
    imag_color = viridis[3]
    
    for model_num in range(1, 5):
        W, _, _, n_model = network_model(model_num, m, phi, alpha, alpha_prime)
        eigenvalues, eigenvectors = linalg.eig(W)
        
        # Sort by magnitude of eigenvalues
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Plot top 6 eigenvectors
        fig, axes = plt.subplots(3, 2, figsize=(7, 10))
        fig.suptitle(f'{model_names[model_num-1]} - Top 6 Eigenvectors', fontsize=14)
        
        for i, ax in enumerate(axes.flat):
            if i < len(eigenvalues):
                # Plot real and imaginary parts
                real_part = np.real(eigenvectors[:, i])
                imag_part = np.imag(eigenvectors[:, i])
                
                # For balanced ring model, plot only the first m components
                if model_num == 4:
                    real_part = real_part[:m]
                    imag_part = imag_part[:m]
                
                ax.plot(phi, real_part, color=real_color, label='Real part', linewidth=2)
                ax.plot(phi, imag_part, color=imag_color, label='Imaginary part', linewidth=2)
                
                # Add eigenvalue information with scientific notation for small values
                eigenvalue = eigenvalues[i]
                if abs(eigenvalue) < 1e-3:
                    title = f'λ = {eigenvalue:.2e}'
                else:
                    title = f'λ = {eigenvalue:.3f}'
                ax.set_title(title, fontsize=12)
                
                ax.set_xlabel(r'Preferred Orientation $\phi$ (rad)', fontsize=10)
                ax.set_ylabel('Eigenvector Component', fontsize=10)
                ax.grid(True, alpha=0.3)
                # ax.legend()
                
                # Set x-axis ticks
                x_ticks = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
                x_tick_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_tick_labels)
                ax.set_xlim(0, 2*np.pi)
        
        plt.tight_layout()
        figs.append(fig)
    
    return figs


### MAIN ###
if __name__ == "__main__":
    
    # COrientation grid and input
    phi = np.linspace(0, 2*np.pi, m, endpoint=False)
    h_theta = stimulus_encoding(phi, theta, kappa)
    
    # Simulate all models
    responses = []
    output_weights = []
    weight_matrices = []
    
    for model_num in range(1, 5):
        r, C, W = simulate_model(model_num, phi, h_theta, tau, dt, num_time_steps)
        responses.append(r)
        output_weights.append(C)
        weight_matrices.append(W)
    
    # Generate and save plots
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    fig1, fig2 = plot_model_responses(responses, output_weights, phi, theta, save_both=True)

    fig3 = plot_comparison_at_60ms(responses, output_weights, phi, theta)
    fig3.savefig(os.path.join(results_dir, 'model_comparison_60ms.png'), dpi=300, bbox_inches='tight')
    
    fig4 = plot_decoding_errors(phi, h_theta, theta, tau, dt, num_time_steps, sigma)
    fig4.savefig(os.path.join(results_dir, 'decoding_error_comparison.png'), dpi=300, bbox_inches='tight')

    eigenvalue_figs = plot_eigenvalues(phi, alpha, alpha_prime)
    for i, fig in enumerate(eigenvalue_figs, 1):
        fig.savefig(os.path.join(results_dir, f'eigenvalues_model_{i}.png'), dpi=300, bbox_inches='tight')
    
    eigenvector_figs = plot_eigenvectors(phi, alpha, alpha_prime)
    for i, fig in enumerate(eigenvector_figs, 1):
        fig.savefig(os.path.join(results_dir, f'eigenvectors_model_{i}.png'), dpi=300, bbox_inches='tight')
