"""
Statistical distribution functions using the incomplete beta function.
"""

import math
from typing import Optional, Union

import torch
from torch import inf, nan, Tensor
from torch.distributions import StudentT as _StudentT

from .betainc import betainc


__all__ = ["StudentT"]


def _cdf_t(x, df, loc=0.0, scale=1.0):
    """
    Internal implementation of Student's t-distribution CDF.
    
    Uses the incomplete beta function to compute the CDF.
    Fully differentiable with respect to all parameters.
    """
    # Convert inputs to tensors if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(df, torch.Tensor):
        df = torch.tensor(df)
    if not isinstance(loc, torch.Tensor):
        loc = torch.tensor(loc)
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale)
    
    # Ensure all tensors have the same dtype and device as x
    df, loc, scale = [t.to(x) for t in (df, loc, scale)]
    
    # Standardize: compute t-statistic
    t = (x - loc) / scale
    
    # Compute the argument for the incomplete beta function
    x_val = df / (df + t.pow(2))
    
    # Compute the incomplete beta function
    prob = betainc(
        df / 2.0,
        torch.full_like(df, 0.5),
        x_val
    )
    
    # Apply the appropriate formula based on the sign of t
    return torch.where(t > 0, 1.0 - 0.5 * prob, 0.5 * prob)


class StudentT(_StudentT):

    def cdf(self, value):
        """
        Compute the cumulative distribution function (CDF).
        
        This method is fully differentiable with respect to both the input value
        and the distribution parameters (df, loc, scale).
        
        Args:
            value (Tensor): The value(s) at which to evaluate the CDF.
            
        Returns:
            Tensor: The CDF value(s).
            
        Example::
        
            >>> dist = StudentT(df=torch.tensor(5.0))
            >>> x = torch.tensor([0.0, 1.0, 2.0])
            >>> dist.cdf(x)
            tensor([0.5000, 0.8182, 0.9489])
        """
        if self._validate_args:
            self._validate_sample(value)
        return _cdf_t(value, self.df, self.loc, self.scale)