"""
Example demonstrating the StudentT distribution with differentiable CDF.

This example shows how to use the StudentT class from torch_betainc,
which provides a differentiable CDF method using the incomplete beta function.
"""

import torch
import matplotlib.pyplot as plt
from torch_betainc import StudentT


def example_basic_usage():
    """Basic usage of StudentT distribution."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create a Student's t-distribution with df=5
    dist = StudentT(df=torch.tensor(5.0))
    
    # Compute CDF at various points
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    cdf_values = dist.cdf(x)
    
    print(f"x values: {x}")
    print(f"CDF values: {cdf_values}")
    print()


def example_with_parameters():
    """StudentT with custom location and scale."""
    print("=" * 60)
    print("Example 2: Custom Location and Scale")
    print("=" * 60)
    
    # Create a t-distribution with df=10, loc=2, scale=1.5
    dist = StudentT(
        df=torch.tensor(10.0),
        loc=torch.tensor(2.0),
        scale=torch.tensor(1.5)
    )
    
    x = torch.tensor([0.0, 2.0, 4.0])
    cdf_values = dist.cdf(x)
    
    print(f"Distribution: df=10, loc=2, scale=1.5")
    print(f"x values: {x}")
    print(f"CDF values: {cdf_values}")
    print()


def example_gradient_computation():
    """Compute gradients through the CDF."""
    print("=" * 60)
    print("Example 3: Gradient Computation")
    print("=" * 60)
    
    # Create distribution with parameters that require gradients
    df = torch.tensor(5.0, requires_grad=True)
    loc = torch.tensor(0.0, requires_grad=True)
    scale = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(1.0, requires_grad=True)
    
    dist = StudentT(df=df, loc=loc, scale=scale)
    cdf_val = dist.cdf(x)
    
    print(f"CDF value at x=1.0: {cdf_val.item():.6f}")
    
    # Compute gradients
    cdf_val.backward()
    
    print(f"∂CDF/∂x = {x.grad.item():.6f}")
    print(f"∂CDF/∂df = {df.grad.item():.6f}")
    print(f"∂CDF/∂loc = {loc.grad.item():.6f}")
    print(f"∂CDF/∂scale = {scale.grad.item():.6f}")
    print()


def example_batch_computation():
    """Batch computation with multiple distributions."""
    print("=" * 60)
    print("Example 4: Batch Computation")
    print("=" * 60)
    
    # Create multiple distributions with different degrees of freedom
    df = torch.tensor([1.0, 2.0, 5.0, 10.0, 30.0])
    dist = StudentT(df=df)
    
    # Evaluate CDF at x=1 for all distributions
    x = torch.tensor(1.0)
    cdf_values = dist.cdf(x)
    
    print("CDF at x=1.0 for different degrees of freedom:")
    for i, (d, c) in enumerate(zip(df, cdf_values)):
        print(f"  df={d.item():.1f}: CDF={c.item():.6f}")
    print()


def example_comparison_with_pytorch():
    """Compare with PyTorch's built-in StudentT."""
    print("=" * 60)
    print("Example 5: Comparison with PyTorch's StudentT")
    print("=" * 60)
    
    from torch.distributions import StudentT as PyTorchStudentT
    
    df = torch.tensor(5.0)
    x = torch.tensor([0.0, 1.0, 2.0])
    
    # Our implementation
    dist_ours = StudentT(df=df)
    cdf_ours = dist_ours.cdf(x)
    log_prob_ours = dist_ours.log_prob(x)
    
    # PyTorch's implementation
    dist_pytorch = PyTorchStudentT(df=df)
    log_prob_pytorch = dist_pytorch.log_prob(x)
    
    print("Our StudentT:")
    print(f"  CDF: {cdf_ours}")
    print(f"  log_prob: {log_prob_ours}")
    print()
    print("PyTorch's StudentT:")
    print(f"  log_prob: {log_prob_pytorch}")
    print(f"  (Note: PyTorch's StudentT doesn't have a CDF method)")
    print()
    print("log_prob difference:", torch.abs(log_prob_ours - log_prob_pytorch).max().item())
    print()


def example_visualization():
    """Visualize the CDF for different degrees of freedom."""
    print("=" * 60)
    print("Example 6: Visualization")
    print("=" * 60)
    
    # Create a range of x values
    x = torch.linspace(-4, 4, 200)
    
    # Different degrees of freedom
    df_values = [1, 2, 5, 10, 30]
    
    plt.figure(figsize=(12, 5))
    
    # Plot CDFs
    plt.subplot(1, 2, 1)
    for df in df_values:
        dist = StudentT(df=torch.tensor(float(df)))
        cdf = dist.cdf(x)
        plt.plot(x.numpy(), cdf.numpy(), label=f'df={df}')
    
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.title("Student's t-distribution CDF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot PDFs (using exp(log_prob))
    plt.subplot(1, 2, 2)
    for df in df_values:
        dist = StudentT(df=torch.tensor(float(df)))
        pdf = torch.exp(dist.log_prob(x))
        plt.plot(x.numpy(), pdf.numpy(), label=f'df={df}')
    
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.title("Student's t-distribution PDF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('studentt_example.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'studentt_example.png'")
    print()


def example_optimization():
    """Use CDF in an optimization problem."""
    print("=" * 60)
    print("Example 7: Optimization with CDF")
    print("=" * 60)
    
    # Goal: Find the value of x where CDF = 0.95 for df=5
    # (This is essentially finding the 95th percentile)
    
    df = torch.tensor(5.0)
    dist = StudentT(df=df)
    
    # Initial guess
    x = torch.tensor(1.0, requires_grad=True)
    target_cdf = 0.95
    
    # Optimize using gradient descent
    optimizer = torch.optim.Adam([x], lr=0.1)
    
    for i in range(100):
        optimizer.zero_grad()
        cdf_val = dist.cdf(x)
        loss = (cdf_val - target_cdf) ** 2
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iteration {i}: x={x.item():.6f}, CDF={cdf_val.item():.6f}, loss={loss.item():.8f}")
    
    print(f"\nFinal result: x={x.item():.6f} gives CDF={dist.cdf(x).item():.6f}")
    print(f"(The 95th percentile for t(df=5) is approximately 2.015)")
    print()


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_with_parameters()
    example_gradient_computation()
    example_batch_computation()
    example_comparison_with_pytorch()
    example_visualization()
    example_optimization()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
