# torch-betainc

A differentiable implementation of the regularized incomplete beta function for PyTorch, with full gradient support for all parameters.

## Features

- **Fully Differentiable**: Compute gradients with respect to all three parameters (a, b, x)
- **Vectorized**: Supports batched computation with tensor inputs
- **Numerically Stable**: Uses continued fraction expansion with convergence tracking
- **Well-Tested**: Comprehensive test suite with gradient verification
- **Easy to Use**: Simple, intuitive API

## Installation

### From source

```bash
git clone https://github.com/k-onoue/torch-betainc.git
cd torch-betainc
pip install -e .
```

### With development dependencies

```bash
pip install -e ".[dev,examples]"
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

### Student's t-Distribution CDF

```python
import torch
from torch_betainc import cdf_t

# Compute CDF of t-distribution
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
df = torch.tensor(10.0)

cdf = cdf_t(x, df)
print(cdf)  # tensor([0.0366, 0.1718, 0.5000, 0.8282, 0.9634])

# With custom location and scale
cdf = cdf_t(x, df, loc=0.0, scale=2.0)
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

### `cdf_t(x, df, loc=0.0, scale=1.0)`

Compute the cumulative distribution function of Student's t-distribution.

**Parameters:**
- `x` (torch.Tensor): The value(s) at which to evaluate the CDF.
- `df` (torch.Tensor): Degrees of freedom. Must be positive.
- `loc` (torch.Tensor, optional): Location parameter (mean). Default: 0.0.
- `scale` (torch.Tensor, optional): Scale parameter. Must be positive. Default: 1.0.

**Returns:**
- `torch.Tensor`: The CDF value(s)

**Example:**
```python
cdf = cdf_t(torch.tensor(1.0), torch.tensor(5.0))
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

- The function uses iterative approximation with a maximum of 200 iterations
- Convergence is typically achieved in fewer than 20 iterations for most inputs
- Batch processing is efficient due to vectorization
- Double precision (`torch.float64`) is recommended for gradient checking

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
