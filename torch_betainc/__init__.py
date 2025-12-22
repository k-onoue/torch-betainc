"""
torch_betainc: Differentiable incomplete beta function for PyTorch
===================================================================

This package provides a differentiable implementation of the regularized 
incomplete beta function and related statistical distributions for PyTorch.

Main Functions
--------------
- betainc: Regularized incomplete beta function I_x(a, b)
- cdf_t: Cumulative distribution function of Student's t-distribution

Examples
--------
>>> import torch
>>> from torch_betainc import betainc, cdf_t
>>> 
>>> # Compute incomplete beta function
>>> a = torch.tensor(2.0, requires_grad=True)
>>> b = torch.tensor(3.0, requires_grad=True)
>>> x = torch.tensor(0.5, requires_grad=True)
>>> result = betainc(a, b, x)
>>> 
>>> # Compute t-distribution CDF
>>> x = torch.tensor(1.0)
>>> df = torch.tensor(5.0)
>>> cdf = cdf_t(x, df)

Credits
-------
Based on the implementation by Arthur Zwaenepoel:
https://github.com/arzwa/IncBetaDer
"""

__version__ = "0.1.0"
__author__ = "Keisuke Onoue"
__credits__ = "Arthur Zwaenepoel"

from .betainc import betainc
from .distributions import cdf_t

__all__ = ["betainc", "cdf_t"]
