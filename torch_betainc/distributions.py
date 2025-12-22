"""
Statistical distribution functions using the incomplete beta function.
"""

import torch
from .betainc import betainc


def cdf_t(x, df, loc=0.0, scale=1.0):
    """
    Compute the cumulative distribution function (CDF) of Student's t-distribution.
    
    This function is fully differentiable with respect to all parameters: x, df, loc, and scale.
    
    The Student's t-distribution is defined by its degrees of freedom (df), location (loc),
    and scale parameters. The CDF gives the probability that a random variable from this
    distribution is less than or equal to x.
    
    Args:
        x (float or torch.Tensor): The value(s) at which to evaluate the CDF.
        df (float or torch.Tensor): Degrees of freedom. Must be positive.
        loc (float or torch.Tensor, optional): Location parameter (mean). Default: 0.0.
        scale (float or torch.Tensor, optional): Scale parameter (standard deviation). 
            Must be positive. Default: 1.0.
            
    Returns:
        torch.Tensor: The CDF value(s). The output shape is determined by broadcasting
            the input shapes.
            
    Examples:
        >>> import torch
        >>> from torch_betainc import cdf_t
        >>> 
        >>> # Single value
        >>> x = torch.tensor(0.0)
        >>> df = torch.tensor(5.0)
        >>> result = cdf_t(x, df)
        >>> print(result)
        tensor(0.5000)
        >>> 
        >>> # Batch computation
        >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> df = torch.tensor(10.0)
        >>> result = cdf_t(x, df)
        >>> print(result)
        tensor([0.0366, 0.1718, 0.5000, 0.8282, 0.9634])
        >>> 
        >>> # With custom location and scale
        >>> x = torch.tensor(5.0)
        >>> df = torch.tensor(10.0)
        >>> loc = torch.tensor(3.0)
        >>> scale = torch.tensor(2.0)
        >>> result = cdf_t(x, df, loc, scale)
        >>> print(result)
        tensor(0.8282)
        >>> 
        >>> # Gradient computation
        >>> x = torch.tensor(1.0, requires_grad=True)
        >>> df = torch.tensor(5.0, requires_grad=True)
        >>> result = cdf_t(x, df)
        >>> result.backward()
        >>> print(f"∂CDF/∂x = {x.grad}")
        >>> print(f"∂CDF/∂df = {df.grad}")
        
    Notes:
        The relationship between the t-distribution CDF and the incomplete beta function is:
        
        For t = (x - loc) / scale,
        CDF(x) = 1 - 0.5 * I_{df/(df+t²)}(df/2, 1/2)  if t > 0
        CDF(x) = 0.5 * I_{df/(df+t²)}(df/2, 1/2)      if t ≤ 0
        
        where I_x(a, b) is the regularized incomplete beta function.
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
