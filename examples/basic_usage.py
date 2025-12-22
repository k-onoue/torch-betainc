"""
Basic usage examples for torch_betainc package.

This script demonstrates how to use the betainc and cdf_t functions
for computing the incomplete beta function and t-distribution CDF.
"""

import torch
from torch_betainc import betainc, cdf_t


def example_betainc_single():
    """Example: Computing incomplete beta function for single values."""
    print("=" * 60)
    print("Example 1: Single Value Computation")
    print("=" * 60)
    
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    x = torch.tensor(0.5, requires_grad=True)
    
    result = betainc(a, b, x)
    print(f"betainc(a={a.item()}, b={b.item()}, x={x.item()}) = {result.item():.6f}")
    
    # Compute gradients
    result.backward()
    print(f"\nGradients:")
    print(f"  ∂I/∂a = {a.grad.item():.6f}")
    print(f"  ∂I/∂b = {b.grad.item():.6f}")
    print(f"  ∂I/∂x = {x.grad.item():.6f}")
    print()


def example_betainc_batch():
    """Example: Batch computation with tensors."""
    print("=" * 60)
    print("Example 2: Batch Computation")
    print("=" * 60)
    
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    x = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    
    result = betainc(a, b, x)
    
    print("Batch computation:")
    for i in range(len(a)):
        print(f"  betainc(a={a[i].item()}, b={b[i].item()}, x={x[i].item()}) = {result[i].item():.6f}")
    print()


def example_betainc_edge_cases():
    """Example: Edge cases (x=0, x=1)."""
    print("=" * 60)
    print("Example 3: Edge Cases")
    print("=" * 60)
    
    a = torch.tensor(2.0)
    b = torch.tensor(3.0)
    
    # x = 0 should give 0
    x0 = torch.tensor(0.0)
    result0 = betainc(a, b, x0)
    print(f"betainc(a={a.item()}, b={b.item()}, x=0.0) = {result0.item():.6f}")
    
    # x = 1 should give 1
    x1 = torch.tensor(1.0)
    result1 = betainc(a, b, x1)
    print(f"betainc(a={a.item()}, b={b.item()}, x=1.0) = {result1.item():.6f}")
    print()


def example_cdf_t_single():
    """Example: Computing t-distribution CDF."""
    print("=" * 60)
    print("Example 4: Student's t-Distribution CDF")
    print("=" * 60)
    
    x = torch.tensor(0.0, requires_grad=True)
    df = torch.tensor(5.0, requires_grad=True)
    
    result = cdf_t(x, df)
    print(f"cdf_t(x={x.item()}, df={df.item()}) = {result.item():.6f}")
    print("(Should be 0.5 at x=0 for symmetric distribution)")
    
    # Compute gradients
    result.backward()
    print(f"\nGradients:")
    print(f"  ∂CDF/∂x = {x.grad.item():.6f}")
    print(f"  ∂CDF/∂df = {df.grad.item():.6f}")
    print()


def example_cdf_t_batch():
    """Example: Batch computation of t-distribution CDF."""
    print("=" * 60)
    print("Example 5: Batch t-Distribution CDF")
    print("=" * 60)
    
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    df = torch.tensor(10.0)
    
    result = cdf_t(x, df)
    
    print(f"t-distribution CDF with df={df.item()}:")
    for i in range(len(x)):
        print(f"  CDF(x={x[i].item():5.1f}) = {result[i].item():.6f}")
    print()


def example_cdf_t_with_location_scale():
    """Example: t-distribution with custom location and scale."""
    print("=" * 60)
    print("Example 6: t-Distribution with Location and Scale")
    print("=" * 60)
    
    x = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    df = torch.tensor(10.0)
    loc = torch.tensor(5.0)  # Mean
    scale = torch.tensor(2.0)  # Standard deviation
    
    result = cdf_t(x, df, loc, scale)
    
    print(f"t-distribution CDF with df={df.item()}, loc={loc.item()}, scale={scale.item()}:")
    for i in range(len(x)):
        print(f"  CDF(x={x[i].item():5.1f}) = {result[i].item():.6f}")
    print()


def example_broadcasting():
    """Example: Broadcasting with different tensor shapes."""
    print("=" * 60)
    print("Example 7: Broadcasting")
    print("=" * 60)
    
    # Shape (3, 1)
    a = torch.tensor([[1.0], [2.0], [3.0]])
    # Shape (1, 4)
    b = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # Shape (1,)
    x = torch.tensor([0.5])
    
    result = betainc(a, b, x)
    
    print(f"Input shapes: a={a.shape}, b={b.shape}, x={x.shape}")
    print(f"Output shape: {result.shape}")
    print(f"\nResult:\n{result}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("torch_betainc: Basic Usage Examples")
    print("=" * 60 + "\n")
    
    example_betainc_single()
    example_betainc_batch()
    example_betainc_edge_cases()
    example_cdf_t_single()
    example_cdf_t_batch()
    example_cdf_t_with_location_scale()
    example_broadcasting()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
