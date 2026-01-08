# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-08

### Added
- **StudentT Distribution Class**: New `StudentT` class that extends PyTorch's `StudentT` with a differentiable `cdf()` method
- Comprehensive test suite for `StudentT` class (16 new tests)
- Example script: `studentt_cdf_example.py` demonstrating StudentT usage with CDF
- Full documentation for `StudentT` class in README

### Changed
- **Breaking**: Removed `cdf_t` from public API (now internal as `_cdf_t`)
- Simplified public API to only export `betainc` and `StudentT`
- Updated documentation to focus on `StudentT` class instead of standalone `cdf_t` function

### Implementation
- `StudentT` inherits from `torch.distributions.StudentT` for maximum compatibility
- Maintains all PyTorch distribution methods (`sample`, `rsample`, `log_prob`, `entropy`)
- CDF method is fully differentiable with respect to value and all distribution parameters

## [0.1.0] - 2025-12-22

### Added
- Initial release of torch-betainc
- Differentiable regularized incomplete beta function (`betainc`)
- Student's t-distribution CDF (`cdf_t`)
- Full gradient support for all parameters
- Configurable precision parameters (`epsilon`, `min_approx`, `max_approx`)
- Comprehensive test suite (29 tests)
- Example scripts:
  - `basic_usage.py` - Basic usage examples
  - `gradient_verification.py` - Gradient verification
  - `custom_precision.py` - Custom precision settings
- Visualization notebook for exploring function behavior
- Full vectorization for batch processing
- Improved convergence for gradient smoothness at extreme values

### Features
- **Fully Differentiable**: Compute gradients with respect to all parameters (a, b, x)
- **Vectorized**: Supports batched computation with tensor inputs
- **Numerically Stable**: Uses continued fraction expansion with convergence tracking
- **Configurable**: Adjustable precision parameters for accuracy/performance trade-off
- **Well-Tested**: Comprehensive test suite with gradient verification

### Credits
Based on the implementation by Arthur Zwaenepoel (https://github.com/arzwa/IncBetaDer)

[0.1.0]: https://github.com/k-onoue/torch-betainc/releases/tag/v0.1.0
