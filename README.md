# torch-betainc

[![PyPI version](https://badge.fury.io/py/torch-betainc.svg)](https://pypi.org/project/torch-betainc/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An implementation of the regularized incomplete beta function for PyTorch, with full gradient support for all parameters.

## Features

- **Fully Differentiable**: Compute gradients with respect to all parameters (a, b, x)
- **Vectorized**: Supports batched computation with tensor inputs
- **Numerically Stable**: Uses continued fraction expansion with convergence tracking
- **Configurable Precision**: Adjustable parameters for accuracy/performance trade-off
- **Well-Tested**: Comprehensive test suite with gradient verification
- **Easy to Use**: Simple, intuitive API

## Installation

### From PyPI (recommended)

```bash
pip install torch-betainc
```

### From source

```bash
git clone https://github.com/k-onoue/torch-betainc.git
cd torch-betainc
pip install -e .
```

### With optional dependencies

```bash
# For development (testing)
pip install torch-betainc[dev]

# For examples (matplotlib, seaborn)
pip install torch-betainc[examples]

# For notebooks (jupyter, visualization)
pip install torch-betainc[notebook]

# All optional dependencies
pip install torch-betainc[dev,examples,notebook]
```

## Quick Start

### Incomplete Beta Function

```python
import torch
from torch_betainc import betainc

# Single values
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(0.5, requires_grad=True)

result = betainc(a, b, x)
print(result)  # tensor(0.6875, grad_fn=<BetaincBackward>)

# Compute gradients
result.backward()
print(f"∂I/∂a = {a.grad}")
print(f"∂I/∂b = {b.grad}")
print(f"∂I/∂x = {x.grad}")
```

### Batch Computation

```python
import torch
from torch_betainc import betainc

# Batch computation
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([1.0, 2.0, 3.0])
x = torch.tensor([0.3, 0.5, 0.7])

result = betainc(a, b, x)
print(result)  # tensor([0.3000, 0.5000, 0.7840])
```

### StudentT Distribution Class

```python
import torch
from torch_betainc import StudentT

# Create a Student's t-distribution
dist = StudentT(df=torch.tensor(5.0))

# Sample from the distribution
samples = dist.sample((1000,))

# Compute CDF (differentiable!)
x = torch.tensor([0.0, 1.0, 2.0])
cdf = dist.cdf(x)
print(cdf)  # tensor([0.5000, 0.8182, 0.9489])

# Compute log probability
log_prob = dist.log_prob(x)

# Compute gradients through CDF
x_grad = torch.tensor(1.0, requires_grad=True)
df_grad = torch.tensor(5.0, requires_grad=True)
dist_grad = StudentT(df=df_grad)
cdf_val = dist_grad.cdf(x_grad)
cdf_val.backward()
print(f"∂CDF/∂x = {x_grad.grad}")
print(f"∂CDF/∂df = {df_grad.grad}")
```

## API Reference

### `betainc(a, b, x, epsilon=1e-14, min_approx=3, max_approx=500)`

Compute the regularized incomplete beta function I_x(a, b).

**Parameters:**
- `a` (torch.Tensor): First shape parameter. Must be positive.
- `b` (torch.Tensor): Second shape parameter. Must be positive.
- `x` (torch.Tensor): Upper limit of integration. Must be in [0, 1].
- `epsilon` (float, optional): Convergence threshold. Default: 1e-14.
- `min_approx` (int, optional): Minimum iterations before checking convergence. Default: 3.
- `max_approx` (int, optional): Maximum iterations for continued fraction. Default: 500.

**Returns:**
- `torch.Tensor`: The value of I_x(a, b)

**Examples:**
```python
# Standard usage
result = betainc(torch.tensor(2.0), torch.tensor(3.0), torch.tensor(0.5))

# Custom precision for faster computation
result = betainc(a, b, x, epsilon=1e-12, max_approx=200)
```

### `StudentT(df, loc=0.0, scale=1.0, validate_args=None)`

Student's t-distribution class with differentiable CDF method.

This class extends PyTorch's distribution interface and provides all standard methods (`sample`, `rsample`, `log_prob`, `entropy`) plus a differentiable `cdf` method.

**Parameters:**
- `df` (float or torch.Tensor): Degrees of freedom. Must be positive.
- `loc` (float or torch.Tensor, optional): Location parameter (mean). Default: 0.0.
- `scale` (float or torch.Tensor, optional): Scale parameter. Must be positive. Default: 1.0.
- `validate_args` (bool, optional): Whether to validate arguments. Default: None.

**Methods:**
- `sample(sample_shape)`: Generate samples from the distribution.
- `rsample(sample_shape)`: Generate reparameterized samples (supports gradients).
- `log_prob(value)`: Compute log probability density.
- `cdf(value)`: Compute cumulative distribution function (differentiable).
- `entropy()`: Compute entropy of the distribution.

**Properties:**
- `mean`: Mean of the distribution (undefined for df ≤ 1).
- `mode`: Mode of the distribution (equals loc).
- `variance`: Variance of the distribution (undefined for df ≤ 1, infinite for 1 < df ≤ 2).

**Example:**
```python
# Create distribution
dist = StudentT(df=torch.tensor(5.0), loc=torch.tensor(0.0), scale=torch.tensor(1.0))

# Use like any PyTorch distribution
samples = dist.rsample((100,))
log_probs = dist.log_prob(samples)

# Compute differentiable CDF
x = torch.tensor(1.0, requires_grad=True)
cdf_val = dist.cdf(x)
cdf_val.backward()  # Gradients flow through CDF!
```

## Examples

The `examples/` directory contains several demonstration scripts:

### Basic Usage

```bash
python examples/basic_usage.py
```

This script demonstrates:
- Single value computation
- Batch processing
- Edge cases
- Gradient computation
- Broadcasting

### Gradient Verification

```bash
python examples/gradient_verification.py
```

This script visually compares analytical gradients (from the custom autograd implementation) with numerical gradients (from finite differences) to verify correctness.

### StudentT Distribution with CDF

```bash
python examples/studentt_cdf_example.py
```

This script demonstrates:
- Creating StudentT distributions
- Computing differentiable CDF values
- Gradient computation through CDF
- Batch computation with multiple distributions
- Comparison with PyTorch's built-in StudentT
- Visualization of CDF and PDF
- Using CDF in optimization problems

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=torch_betainc --cov-report=html
```

## Mathematical Background

### Regularized Incomplete Beta Function

The regularized incomplete beta function is defined as:

```
I_x(a, b) = B(x; a, b) / B(a, b)
```

where:
- `B(x; a, b)` is the incomplete beta function
- `B(a, b)` is the complete beta function

This implementation uses a continued fraction expansion for numerical computation, with automatic switching based on the symmetry relation `I_x(a, b) = 1 - I_{1-x}(b, a)` to improve numerical stability.

### Student's t-Distribution CDF

The CDF of Student's t-distribution is computed using the incomplete beta function:

```
For t = (x - loc) / scale,
CDF(x) = 1 - 0.5 * I_{df/(df+t²)}(df/2, 1/2)  if t > 0
CDF(x) = 0.5 * I_{df/(df+t²)}(df/2, 1/2)      if t ≤ 0
```

## Implementation Details

- **Continued Fraction**: Uses a modified Lentz algorithm for the continued fraction expansion
- **Convergence**: Tracks convergence per element in batched computations
- **Numerical Stability**: Implements safeguards against division by zero and uses the symmetry relation
- **Gradients**: Computes analytical gradients for all parameters using custom backward pass

## Performance Considerations

- The function uses iterative approximation with a default maximum of 500 iterations
- Convergence is typically achieved in fewer than 20 iterations for most inputs
- Batch processing is efficient due to vectorization
- Double precision (`torch.float64`) is recommended for gradient checking
- Precision parameters (`epsilon`, `max_approx`) can be customized for performance tuning

## Credits

This implementation is based on the work by **Arthur Zwaenepoel**:
- GitHub: https://github.com/arzwa/IncBetaDer
- Google Scholar: https://scholar.google.com/citations?user=8VSQd34AAAAJ&hl=en

The code has been refactored and extended to:
- Support full vectorization for batch processing
- Include comprehensive documentation and tests
- Add Student's t-distribution CDF
- Fix gradient computation for `gradcheck` compatibility

## License

MIT License - see LICENSE file for details

## Support

For bug reports and feature requests, please open an issue on GitHub.
