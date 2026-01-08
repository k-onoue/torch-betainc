"""
Tests for the StudentT distribution class with CDF method.
"""

import pytest
import torch
import numpy as np
from scipy import stats

from torch_betainc import StudentT


class TestStudentT:
    """Test suite for StudentT distribution."""

    def test_initialization(self):
        """Test that StudentT can be initialized with various parameter types."""
        # Scalar parameters
        dist = StudentT(df=5.0)
        assert dist.df.item() == 5.0
        assert dist.loc.item() == 0.0
        assert dist.scale.item() == 1.0

        # Tensor parameters
        df = torch.tensor([2.0, 5.0, 10.0])
        loc = torch.tensor([0.0, 1.0, -1.0])
        scale = torch.tensor([1.0, 2.0, 0.5])
        dist = StudentT(df, loc, scale)
        assert torch.allclose(dist.df, df)
        assert torch.allclose(dist.loc, loc)
        assert torch.allclose(dist.scale, scale)

    def test_properties(self):
        """Test distribution properties (mean, mode, variance)."""
        df = torch.tensor([1.0, 2.0, 3.0, 5.0])
        loc = torch.tensor([0.0, 1.0, -1.0, 2.0])
        scale = torch.tensor([1.0, 2.0, 0.5, 1.5])
        dist = StudentT(df, loc, scale)

        # Mode should equal loc
        assert torch.allclose(dist.mode, loc)

        # Mean is undefined for df <= 1, equals loc for df > 1
        mean = dist.mean
        assert torch.isnan(mean[0])  # df=1
        assert torch.allclose(mean[1:], loc[1:])

        # Variance
        variance = dist.variance
        assert torch.isnan(variance[0])  # df=1
        assert torch.isinf(variance[1])  # df=2
        # For df > 2: Var = scale^2 * df / (df - 2)
        expected_var_2 = scale[2] ** 2 * df[2] / (df[2] - 2)
        expected_var_3 = scale[3] ** 2 * df[3] / (df[3] - 2)
        assert torch.allclose(variance[2], expected_var_2)
        assert torch.allclose(variance[3], expected_var_3)

    def test_cdf_standard(self):
        """Test CDF against scipy for standard t-distribution."""
        df_values = [1.0, 2.0, 5.0, 10.0, 30.0]
        x_values = [-3.0, -1.0, 0.0, 1.0, 3.0]

        for df in df_values:
            dist = StudentT(df=torch.tensor(df))
            for x in x_values:
                x_tensor = torch.tensor(x)
                cdf_torch = dist.cdf(x_tensor).item()
                cdf_scipy = stats.t.cdf(x, df)
                assert abs(cdf_torch - cdf_scipy) < 1e-5, (
                    f"CDF mismatch at df={df}, x={x}: "
                    f"torch={cdf_torch}, scipy={cdf_scipy}"
                )

    def test_cdf_with_loc_scale(self):
        """Test CDF with non-standard location and scale."""
        df = 5.0
        loc = 2.0
        scale = 1.5
        x_values = [-1.0, 0.0, 2.0, 4.0, 6.0]

        dist = StudentT(df=torch.tensor(df), loc=torch.tensor(loc), scale=torch.tensor(scale))

        for x in x_values:
            x_tensor = torch.tensor(x)
            cdf_torch = dist.cdf(x_tensor).item()
            cdf_scipy = stats.t.cdf(x, df, loc=loc, scale=scale)
            assert abs(cdf_torch - cdf_scipy) < 1e-5, (
                f"CDF mismatch at x={x}: torch={cdf_torch}, scipy={cdf_scipy}"
            )

    def test_cdf_batch(self):
        """Test CDF with batched inputs."""
        df = torch.tensor([2.0, 5.0, 10.0])
        x = torch.tensor([0.0, 1.0, 2.0])

        dist = StudentT(df=df)
        cdf_values = dist.cdf(x)

        # Compare with scipy
        for i in range(len(df)):
            cdf_scipy = stats.t.cdf(x[i].item(), df[i].item())
            assert abs(cdf_values[i].item() - cdf_scipy) < 1e-5

    def test_cdf_gradient_x(self):
        """Test that CDF is differentiable with respect to x."""
        df = torch.tensor(5.0)
        x = torch.tensor(1.0, requires_grad=True)

        dist = StudentT(df=df)
        cdf_val = dist.cdf(x)
        cdf_val.backward()

        # Gradient should exist and be positive (CDF is increasing)
        assert x.grad is not None
        assert x.grad.item() > 0

        # Gradient should approximately equal the PDF
        pdf_val = torch.exp(dist.log_prob(x))
        assert abs(x.grad.item() - pdf_val.item()) < 1e-4

    def test_cdf_gradient_df(self):
        """Test that CDF is differentiable with respect to df."""
        df = torch.tensor(5.0, requires_grad=True)
        x = torch.tensor(1.0)

        dist = StudentT(df=df)
        cdf_val = dist.cdf(x)
        cdf_val.backward()

        # Gradient should exist
        assert df.grad is not None
        # For x > 0, increasing df should increase CDF (approach normal)
        assert df.grad.item() > 0

    def test_cdf_gradient_loc_scale(self):
        """Test that CDF is differentiable with respect to loc and scale."""
        df = torch.tensor(5.0)
        loc = torch.tensor(0.0, requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        x = torch.tensor(1.0)

        dist = StudentT(df=df, loc=loc, scale=scale)
        cdf_val = dist.cdf(x)
        cdf_val.backward()

        # Gradients should exist
        assert loc.grad is not None
        assert scale.grad is not None

        # For x > loc, increasing loc should decrease CDF (shifts distribution right)
        # Think: if we move the distribution to the right, x becomes relatively smaller
        assert loc.grad.item() < 0
        # For x > loc, increasing scale should decrease CDF (spreads distribution)
        assert scale.grad.item() < 0

    def test_cdf_boundary_values(self):
        """Test CDF at boundary values."""
        dist = StudentT(df=torch.tensor(5.0))

        # CDF at 0 should be 0.5 for standard t-distribution
        cdf_zero = dist.cdf(torch.tensor(0.0))
        assert abs(cdf_zero.item() - 0.5) < 1e-6

        # CDF should approach 0 for large negative values
        cdf_neg = dist.cdf(torch.tensor(-10.0))
        assert cdf_neg.item() < 1e-4  # Relaxed tolerance

        # CDF should approach 1 for large positive values
        cdf_pos = dist.cdf(torch.tensor(10.0))
        assert cdf_pos.item() > 1 - 1e-4  # Relaxed tolerance

    def test_sampling(self):
        """Test that sampling works correctly."""
        dist = StudentT(df=torch.tensor(5.0))
        samples = dist.sample((1000,))

        # Check shape
        assert samples.shape == (1000,)

        # Check that samples are finite
        assert torch.all(torch.isfinite(samples))

    def test_log_prob(self):
        """Test log probability computation."""
        df = torch.tensor(5.0)
        dist = StudentT(df=df)
        x = torch.tensor([0.0, 1.0, 2.0])

        log_prob = dist.log_prob(x)

        # Compare with scipy
        for i in range(len(x)):
            log_prob_scipy = stats.t.logpdf(x[i].item(), df.item())
            assert abs(log_prob[i].item() - log_prob_scipy) < 1e-5

    def test_entropy(self):
        """Test entropy computation."""
        df = torch.tensor(5.0)
        dist = StudentT(df=df)

        entropy = dist.entropy()

        # Compare with scipy
        entropy_scipy = stats.t.entropy(df.item())
        assert abs(entropy.item() - entropy_scipy) < 1e-5

    def test_expand(self):
        """Test distribution expansion."""
        dist = StudentT(df=torch.tensor(5.0))
        expanded = dist.expand((3,))

        assert expanded.df.shape == (3,)
        assert torch.all(expanded.df == 5.0)

    def test_broadcasting(self):
        """Test parameter broadcasting."""
        df = torch.tensor([2.0, 5.0, 10.0])
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)

        dist = StudentT(df=df, loc=loc, scale=scale)

        assert dist.df.shape == (3,)
        assert dist.loc.shape == (3,)
        assert dist.scale.shape == (3,)

    def test_dtype_consistency(self):
        """Test that CDF maintains dtype consistency."""
        # Float32
        dist_f32 = StudentT(df=torch.tensor(5.0, dtype=torch.float32))
        x_f32 = torch.tensor(1.0, dtype=torch.float32)
        cdf_f32 = dist_f32.cdf(x_f32)
        assert cdf_f32.dtype == torch.float32

        # Float64
        dist_f64 = StudentT(df=torch.tensor(5.0, dtype=torch.float64))
        x_f64 = torch.tensor(1.0, dtype=torch.float64)
        cdf_f64 = dist_f64.cdf(x_f64)
        assert cdf_f64.dtype == torch.float64

    def test_device_consistency(self):
        """Test that CDF works on different devices."""
        dist = StudentT(df=torch.tensor(5.0))
        x = torch.tensor(1.0)
        cdf = dist.cdf(x)

        assert cdf.device == x.device

        # Test CUDA if available
        if torch.cuda.is_available():
            dist_cuda = StudentT(df=torch.tensor(5.0, device='cuda'))
            x_cuda = torch.tensor(1.0, device='cuda')
            cdf_cuda = dist_cuda.cdf(x_cuda)
            assert cdf_cuda.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
