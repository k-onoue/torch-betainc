"""
torch_betainc: Differentiable incomplete beta function for PyTorch
===================================================================

This package provides a differentiable implementation of the regularized 
incomplete beta function and related statistical distributions for PyTorch.

Main Functions
--------------
- betainc: Regularized incomplete beta function I_x(a, b)
- StudentT: Student's t-distribution with differentiable CDF method

Examples
--------
>>> import torch
>>> from torch_betainc import betainc, StudentT
>>> 
>>> # Compute incomplete beta function
>>> a = torch.tensor(2.0, requires_grad=True)
>>> b = torch.tensor(3.0, requires_grad=True)
>>> x = torch.tensor(0.5, requires_grad=True)
>>> result = betainc(a, b, x)
>>> 
>>> # Compute t-distribution CDF
>>> dist = StudentT(df=torch.tensor(5.0))
>>> x = torch.tensor(1.0)
>>> cdf = dist.cdf(x)

Credits
-------
Based on the implementation by Arthur Zwaenepoel:
https://github.com/arzwa/IncBetaDer
"""

__version__ = "0.2.0"
__author__ = "Keisuke Onoue"
__credits__ = "Arthur Zwaenepoel"

from .betainc import betainc
from .distributions import StudentT

__all__ = ["betainc", "StudentT"]
