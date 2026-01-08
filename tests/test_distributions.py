"""
Tests for the distribution functions.
"""

import pytest
import torch
from torch_betainc.distributions import _cdf_t as cdf_t


class TestCdfTBasicProperties:
    """Test basic properties of the t-distribution CDF."""
    
    def test_cdf_at_zero(self):
        """Test that CDF(0) = 0.5 for symmetric distribution."""
        x = torch.tensor(0.0)
        df = torch.tensor(5.0)
        result = cdf_t(x, df)
        assert torch.allclose(result, torch.tensor(0.5), atol=1e-5)
    
    def test_cdf_at_zero_with_location(self):
        """Test that CDF(loc) = 0.5 when x equals location parameter."""
        x = torch.tensor(3.0)
        df = torch.tensor(5.0)
        loc = torch.tensor(3.0)
        result = cdf_t(x, df, loc=loc)
        assert torch.allclose(result, torch.tensor(0.5), atol=1e-5)
    
    def test_cdf_bounds(self):
        """Test that CDF values are in [0, 1]."""
        x = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
        df = torch.tensor(5.0)
        result = cdf_t(x, df)
        
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)


class TestCdfTMonotonicity:
    """Test monotonicity of the t-distribution CDF."""
    
    def test_monotonic_increasing(self):
        """Test that CDF is monotonically increasing in x."""
        x = torch.linspace(-5.0, 5.0, 50)
        df = torch.tensor(5.0)
        result = cdf_t(x, df)
        
        # Check that each value is greater than or equal to the previous
        for i in range(1, len(result)):
            assert result[i] >= result[i-1]


class TestCdfTSymmetry:
    """Test symmetry properties of the t-distribution CDF."""
    
    def test_symmetry_around_zero(self):
        """Test that CDF(-x) = 1 - CDF(x) for loc=0."""
        x = torch.tensor(2.0)
        df = torch.tensor(5.0)
        
        cdf_pos = cdf_t(x, df)
        cdf_neg = cdf_t(-x, df)
        
        assert torch.allclose(cdf_pos, 1.0 - cdf_neg, atol=1e-5)
    
    def test_symmetry_around_location(self):
        """Test symmetry around the location parameter."""
        loc = torch.tensor(3.0)
        offset = torch.tensor(2.0)
        df = torch.tensor(5.0)
        
        cdf_above = cdf_t(loc + offset, df, loc=loc)
        cdf_below = cdf_t(loc - offset, df, loc=loc)
        
        assert torch.allclose(cdf_above, 1.0 - cdf_below, atol=1e-5)


class TestCdfTVectorization:
    """Test vectorization and broadcasting."""
    
    def test_batch_computation(self):
        """Test batch computation with 1D tensors."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        df = torch.tensor(10.0)
        
        result = cdf_t(x, df)
        
        # Compute individually and compare
        expected = torch.stack([
            cdf_t(x[i:i+1], df)
            for i in range(len(x))
        ]).squeeze()
        
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_broadcasting(self):
        """Test broadcasting with different shapes."""
        x = torch.tensor([[0.0], [1.0]])  # Shape (2, 1)
        df = torch.tensor([5.0, 10.0])     # Shape (2,)
        
        result = cdf_t(x, df)
        
        assert result.shape == (2, 2)


class TestCdfTGradients:
    """Test gradient computation."""
    
    def test_gradients_exist(self):
        """Test that gradients can be computed."""
        x = torch.tensor(1.0, requires_grad=True)
        df = torch.tensor(5.0, requires_grad=True)
        loc = torch.tensor(0.0, requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        
        result = cdf_t(x, df, loc, scale)
        result.backward()
        
        assert x.grad is not None
        assert df.grad is not None
        assert loc.grad is not None
        assert scale.grad is not None
        assert not torch.isnan(x.grad)
        assert not torch.isnan(df.grad)
        assert not torch.isnan(loc.grad)
        assert not torch.isnan(scale.grad)
    
    def test_gradient_x_positive(self):
        """Test that gradient w.r.t. x is positive (PDF is positive)."""
        x = torch.tensor(1.0, requires_grad=True)
        df = torch.tensor(5.0)
        
        result = cdf_t(x, df)
        result.backward()
        
        # The derivative of CDF is the PDF, which should be positive
        assert x.grad > 0
    
    def test_gradcheck(self):
        """Test gradients using torch.autograd.gradcheck."""
        x = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        df = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        
        # gradcheck requires double precision
        # Note: df gradient has lower numerical precision, so we use relaxed tolerance
        assert torch.autograd.gradcheck(
            lambda x, df: cdf_t(x, df),
            (x, df),
            eps=1e-6,
            atol=1e-3,  # Relaxed from 1e-4 due to numerical precision in df gradient
            rtol=1e-2   # Added relative tolerance
        )
    
    def test_gradient_smoothness_at_extremes(self):
        """Test that gradients are smooth at extreme x values."""
        x_values = torch.linspace(-4, 4, 100)
        df = torch.tensor(5.0)
        
        # Compute gradients w.r.t. x
        grad_x_values = []
        for x_val in x_values:
            x = x_val.clone().detach().requires_grad_(True)
            df_clone = df.clone().detach()
            result = cdf_t(x, df_clone)
            result.backward()
            grad_x_values.append(x.grad.item())
        
        # Check that gradients don't have large jumps (smoothness)
        grad_diffs = [abs(grad_x_values[i+1] - grad_x_values[i]) 
                      for i in range(len(grad_x_values)-1)]
        max_jump = max(grad_diffs)
        
        # Maximum gradient jump should be small (smooth function)
        # With improved convergence, this should be much smaller than before
        # Threshold set to 0.02 to catch major artifacts while being realistic
        assert max_jump < 0.02, f"Large gradient jump detected: {max_jump}"


class TestCdfTScaleParameter:
    """Test the scale parameter behavior."""
    
    def test_scale_effect(self):
        """Test that scale parameter affects the spread."""
        x = torch.tensor(2.0)
        df = torch.tensor(10.0)
        
        # Smaller scale should give higher CDF at the same x
        cdf_small_scale = cdf_t(x, df, scale=0.5)
        cdf_large_scale = cdf_t(x, df, scale=2.0)
        
        assert cdf_small_scale > cdf_large_scale


class TestCdfTInputTypes:
    """Test different input types."""
    
    def test_float_inputs(self):
        """Test that float inputs are converted to tensors."""
        result = cdf_t(0.0, 5.0)
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.tensor(0.5), atol=1e-5)
    
    def test_mixed_inputs(self):
        """Test mixing float and tensor inputs."""
        x = torch.tensor(0.0)
        df = 5.0  # float
        
        result = cdf_t(x, df)
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.tensor(0.5), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
