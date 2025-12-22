"""
Tests for the betainc function.
"""

import pytest
import torch
from torch_betainc import betainc


class TestBetaincEdgeCases:
    """Test edge cases for the betainc function."""
    
    def test_x_equals_zero(self):
        """Test that betainc(a, b, 0) = 0."""
        a = torch.tensor(2.0)
        b = torch.tensor(3.0)
        x = torch.tensor(0.0)
        result = betainc(a, b, x)
        assert torch.allclose(result, torch.tensor(0.0), atol=1e-6)
    
    def test_x_equals_one(self):
        """Test that betainc(a, b, 1) = 1."""
        a = torch.tensor(2.0)
        b = torch.tensor(3.0)
        x = torch.tensor(1.0)
        result = betainc(a, b, x)
        assert torch.allclose(result, torch.tensor(1.0), atol=1e-6)
    
    def test_x_below_zero(self):
        """Test that betainc(a, b, x<0) = 0."""
        a = torch.tensor(2.0)
        b = torch.tensor(3.0)
        x = torch.tensor(-0.5)
        result = betainc(a, b, x)
        assert torch.allclose(result, torch.tensor(0.0), atol=1e-6)
    
    def test_x_above_one(self):
        """Test that betainc(a, b, x>1) = 1."""
        a = torch.tensor(2.0)
        b = torch.tensor(3.0)
        x = torch.tensor(1.5)
        result = betainc(a, b, x)
        assert torch.allclose(result, torch.tensor(1.0), atol=1e-6)


class TestBetaincSymmetry:
    """Test symmetry properties of the betainc function."""
    
    def test_symmetry_relation(self):
        """Test that I_x(a, b) = 1 - I_{1-x}(b, a)."""
        a = torch.tensor(2.0)
        b = torch.tensor(3.0)
        x = torch.tensor(0.3)
        
        result1 = betainc(a, b, x)
        result2 = 1.0 - betainc(b, a, 1.0 - x)
        
        assert torch.allclose(result1, result2, atol=1e-5)


class TestBetaincNumericalAccuracy:
    """Test numerical accuracy against known values."""
    
    def test_known_value_1(self):
        """Test betainc(1, 1, 0.5) = 0.5."""
        a = torch.tensor(1.0)
        b = torch.tensor(1.0)
        x = torch.tensor(0.5)
        result = betainc(a, b, x)
        expected = torch.tensor(0.5)
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_known_value_2(self):
        """Test betainc(2, 2, 0.5) = 0.5."""
        a = torch.tensor(2.0)
        b = torch.tensor(2.0)
        x = torch.tensor(0.5)
        result = betainc(a, b, x)
        expected = torch.tensor(0.5)
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_known_value_3(self):
        """Test betainc(2, 3, 0.5) ≈ 0.6875."""
        a = torch.tensor(2.0)
        b = torch.tensor(3.0)
        x = torch.tensor(0.5)
        result = betainc(a, b, x)
        expected = torch.tensor(0.6875)
        assert torch.allclose(result, expected, atol=1e-4)


class TestBetaincVectorization:
    """Test vectorization and broadcasting."""
    
    def test_batch_computation(self):
        """Test batch computation with 1D tensors."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        x = torch.tensor([0.3, 0.5, 0.7])
        
        result = betainc(a, b, x)
        
        # Compute individually and compare
        expected = torch.stack([
            betainc(a[i:i+1], b[i:i+1], x[i:i+1])
            for i in range(len(a))
        ]).squeeze()
        
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_broadcasting(self):
        """Test broadcasting with different shapes."""
        a = torch.tensor([[1.0], [2.0]])  # Shape (2, 1)
        b = torch.tensor([[1.0, 2.0]])     # Shape (1, 2)
        x = torch.tensor(0.5)              # Shape ()
        
        result = betainc(a, b, x)
        
        assert result.shape == (2, 2)
        
        # Check individual values
        assert torch.allclose(result[0, 0], betainc(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.5)))
        assert torch.allclose(result[0, 1], betainc(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.5)))
        assert torch.allclose(result[1, 0], betainc(torch.tensor(2.0), torch.tensor(1.0), torch.tensor(0.5)))
        assert torch.allclose(result[1, 1], betainc(torch.tensor(2.0), torch.tensor(2.0), torch.tensor(0.5)))


class TestBetaincGradients:
    """Test gradient computation."""
    
    def test_gradients_exist(self):
        """Test that gradients can be computed."""
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)
        x = torch.tensor(0.5, requires_grad=True)
        
        result = betainc(a, b, x)
        result.backward()
        
        assert a.grad is not None
        assert b.grad is not None
        assert x.grad is not None
        assert not torch.isnan(a.grad)
        assert not torch.isnan(b.grad)
        assert not torch.isnan(x.grad)
    
    def test_gradient_x_positive(self):
        """Test that gradient w.r.t. x is positive (CDF is monotonic)."""
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)
        x = torch.tensor(0.5, requires_grad=True)
        
        result = betainc(a, b, x)
        result.backward()
        
        # The derivative should be positive (monotonic increasing)
        assert x.grad > 0
    
    def test_gradcheck(self):
        """Test gradients using torch.autograd.gradcheck."""
        a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        x = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        
        # gradcheck requires double precision
        assert torch.autograd.gradcheck(
            lambda a, b, x: betainc(a, b, x),
            (a, b, x),
            eps=1e-6,
            atol=1e-4
        )


class TestBetaincMonotonicity:
    """Test monotonicity properties."""
    
    def test_monotonic_in_x(self):
        """Test that betainc is monotonically increasing in x."""
        a = torch.tensor(2.0)
        b = torch.tensor(3.0)
        x_values = torch.linspace(0.1, 0.9, 10)
        
        results = betainc(a, b, x_values)
        
        # Check that each value is greater than or equal to the previous
        for i in range(1, len(results)):
            assert results[i] >= results[i-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
