"""
Example demonstrating custom precision settings for betainc.

This shows how to use the epsilon, min_approx, and max_approx parameters
to control the trade-off between accuracy and performance.
"""

import torch
import time
from torch_betainc import betainc

print("=" * 60)
print("Custom Precision Settings Example")
print("=" * 60)

# Test parameters
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(0.5, requires_grad=True)

print("\n1. Default (high precision) settings:")
print("   epsilon=1e-14, max_approx=500")
result_default = betainc(a, b, x)
print(f"   Result: {result_default.item():.10f}")

# Reset gradients
a.grad = None
b.grad = None
x.grad = None

print("\n2. Original (faster) settings:")
print("   epsilon=1e-12, max_approx=200")
result_original = betainc(a, b, x, epsilon=1e-12, max_approx=200)
print(f"   Result: {result_original.item():.10f}")
print(f"   Difference: {abs(result_default.item() - result_original.item()):.2e}")

# Reset gradients
a.grad = None
b.grad = None
x.grad = None

print("\n3. Ultra-high precision settings:")
print("   epsilon=1e-16, max_approx=1000")
result_ultra = betainc(a, b, x, epsilon=1e-16, max_approx=1000)
print(f"   Result: {result_ultra.item():.10f}")
print(f"   Difference from default: {abs(result_default.item() - result_ultra.item()):.2e}")

print("\n" + "=" * 60)
print("Performance Comparison")
print("=" * 60)

# Batch test
torch.manual_seed(42)
a_batch = torch.randn(1000).abs() + 2
b_batch = torch.randn(1000).abs() + 2
x_batch = torch.rand(1000)

# Benchmark default settings
start = time.time()
for _ in range(100):
    result = betainc(a_batch, b_batch, x_batch)
time_default = (time.time() - start) / 100

# Benchmark original settings
start = time.time()
for _ in range(100):
    result = betainc(a_batch, b_batch, x_batch, epsilon=1e-12, max_approx=200)
time_original = (time.time() - start) / 100

print(f"\nDefault settings (epsilon=1e-14, max_approx=500):")
print(f"  Time per batch: {time_default*1000:.2f} ms")

print(f"\nOriginal settings (epsilon=1e-12, max_approx=200):")
print(f"  Time per batch: {time_original*1000:.2f} ms")

print(f"\nSpeedup with original settings: {time_default/time_original:.2f}x")

print("\n" + "=" * 60)
print("Gradient Computation with Custom Settings")
print("=" * 60)

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(0.5, requires_grad=True)

# Compute with default settings
result = betainc(a, b, x)
result.backward()

print("\nDefault settings gradients:")
print(f"  ∂I/∂a = {a.grad.item():.10f}")
print(f"  ∂I/∂b = {b.grad.item():.10f}")
print(f"  ∂I/∂x = {x.grad.item():.10f}")

# Reset and compute with original settings
a.grad = None
b.grad = None
x.grad = None

result_orig = betainc(a, b, x, epsilon=1e-12, max_approx=200)
result_orig.backward()

print("\nOriginal settings gradients:")
print(f"  ∂I/∂a = {a.grad.item():.10f}")
print(f"  ∂I/∂b = {b.grad.item():.10f}")
print(f"  ∂I/∂x = {x.grad.item():.10f}")

print("\n" + "=" * 60)
print("Recommendation:")
print("=" * 60)
print("""
- Use default settings (epsilon=1e-14, max_approx=500) for:
  * Applications requiring high precision gradients
  * Optimization near extreme values
  * Critical numerical computations

- Use original settings (epsilon=1e-12, max_approx=200) for:
  * Faster computation when extreme precision is not needed
  * Batch processing of many samples
  * Applications away from boundary values (x ≈ 0 or x ≈ 1)
""")
