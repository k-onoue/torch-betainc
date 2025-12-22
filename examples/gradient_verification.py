"""
Visual gradient verification for torch_betainc.

This script compares analytical gradients (from the custom autograd implementation)
with numerical gradients (from finite differences) to verify correctness.

Adapted from the original verification code by Arthur Zwaenepoel.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch_betainc import cdf_t


# Set a consistent data type and a small epsilon for finite differences
DTYPE = torch.float64
EPS = 1e-6


def verify_gradients(params, param_key_to_check):
    """
    Compute and return both analytical and numerical gradients for a specific parameter.
    
    Args:
        params (dict): Dictionary of all required parameters (x, df, loc, scale).
        param_key_to_check (str): The key in the params dict for which to compute the gradient.
        
    Returns:
        A tuple containing (analytical_gradient, numerical_gradient).
    """
    # --- Analytical Gradient (from custom autograd function) ---
    
    # Create tensor copies of all parameters, enabling grad for the target parameter
    t_params = {k: torch.tensor(v, dtype=DTYPE, requires_grad=(k == param_key_to_check)) 
                for k, v in params.items()}
    
    # Run the forward pass
    cdf_val = cdf_t(**t_params)
    
    # Run the backward pass to compute gradients
    cdf_val.backward()
    
    # Extract the computed gradient
    analytical_grad = t_params[param_key_to_check].grad.item()
    
    # --- Numerical Gradient (using the finite difference method) ---
    
    # Create two versions of the parameters, one with a small positive
    # perturbation (eps) and one with a negative one.
    params_plus = params.copy()
    params_plus[param_key_to_check] += EPS
    t_params_plus = {k: torch.tensor(v, dtype=DTYPE) for k, v in params_plus.items()}

    params_minus = params.copy()
    params_minus[param_key_to_check] -= EPS
    t_params_minus = {k: torch.tensor(v, dtype=DTYPE) for k, v in params_minus.items()}

    # Compute the CDF at these two perturbed points
    cdf_plus = cdf_t(**t_params_plus)
    cdf_minus = cdf_t(**t_params_minus)
    
    # The centered finite difference formula: (f(x+h) - f(x-h)) / 2h
    numerical_grad = (cdf_plus - cdf_minus) / (2 * EPS)

    return analytical_grad, numerical_grad.item()


def plot_gradient_comparison(param_to_vary, fixed_params, param_range):
    """
    Generate and display plots comparing analytical and numerical gradients
    over a range of values for a single parameter.
    """
    print(f"\n📊 Generating plot for parameter: '{param_to_vary}'...")
    
    analytical_grads = []
    numerical_grads = []
    
    # Iterate over the specified range, computing both gradients at each point
    for val in param_range:
        current_params = fixed_params.copy()
        current_params[param_to_vary] = val.item()
        
        ag, ng = verify_gradients(current_params, param_to_vary)
        analytical_grads.append(ag)
        numerical_grads.append(ng)
        
    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    title = f"Gradient Verification for '{param_to_vary}' (other parameters fixed)"
    fig.suptitle(title, fontsize=16)

    # Plot 1: Direct comparison of the two gradient calculation methods
    axes[0].plot(param_range.numpy(), analytical_grads, label='Analytical Gradient (Custom Autograd)', lw=2.5, c='royalblue')
    axes[0].plot(param_range.numpy(), numerical_grads, label='Numerical Gradient (Finite Diff.)', ls='--', c='darkorange', lw=2)
    axes[0].set_xlabel(f"Value of '{param_to_vary}'")
    axes[0].set_ylabel("Gradient Value")
    axes[0].set_title(f"∂(CDF) / ∂({param_to_vary})")
    axes[0].legend()
    axes[0].grid(True, linestyle=':')
    
    # Plot 2: Absolute error between the two methods on a log scale
    abs_error = torch.tensor([abs(a - n) for a, n in zip(analytical_grads, numerical_grads)])
    axes[1].plot(param_range.numpy(), abs_error.numpy(), c='crimson', lw=2)
    axes[1].set_yscale('log')
    axes[1].set_xlabel(f"Value of '{param_to_vary}'")
    axes[1].set_ylabel("Absolute Error (log scale)")
    axes[1].set_title("Error between Analytical and Numerical")
    axes[1].grid(True, which='both', linestyle=':')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Print summary statistics
    max_error = abs_error.max().item()
    mean_error = abs_error.mean().item()
    print(f"  Max absolute error: {max_error:.2e}")
    print(f"  Mean absolute error: {mean_error:.2e}")


def main():
    """Run gradient verification for all parameters."""
    # Set a professional plotting style
    sns.set_theme(style="whitegrid")

    # Define a base set of parameters that will remain fixed while we vary one at a time
    base_params = {
        'x': 1.5,
        'df': 5.0,
        'loc': 0.5,
        'scale': 1.2
    }
    
    print("=" * 60)
    print("Visual Gradient Verification for torch_betainc")
    print("=" * 60)
    print(f"Base parameters (fixed): {base_params}")
    print("=" * 60)

    # --- Generate a plot for each parameter ---

    # 1. Varying 'x' (the point at which the CDF is evaluated)
    x_range = torch.linspace(-3.0, 4.0, 100, dtype=DTYPE)
    plot_gradient_comparison('x', base_params, x_range)
    
    # 2. Varying 'df' (degrees of freedom)
    df_range = torch.linspace(2.0, 30.0, 100, dtype=DTYPE)
    plot_gradient_comparison('df', base_params, df_range)
    
    # 3. Varying 'loc' (the location or mean of the distribution)
    loc_range = torch.linspace(-1.0, 2.0, 100, dtype=DTYPE)
    plot_gradient_comparison('loc', base_params, loc_range)
    
    # 4. Varying 'scale' (the standard deviation of the distribution)
    scale_range = torch.linspace(0.5, 3.0, 100, dtype=DTYPE)
    plot_gradient_comparison('scale', base_params, scale_range)
    
    print("\n" + "=" * 60)
    print("Gradient verification completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
