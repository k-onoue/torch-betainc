"""
Differentiable regularized incomplete beta function for PyTorch.

This implementation is based on the code shared by Arthur Zwaenepoel on GitHub.
See https://github.com/arzwa/IncBetaDer for details.
His Google Scholar profile: https://scholar.google.com/citations?user=8VSQd34AAAAJ&hl=en

This version has been vectorized to support tensor inputs for batch processing
and includes fixes in the backward pass for gradcheck compatibility.
"""

import torch
from torch.autograd import Function

from ._utils import (
    _beta,
    _Kfun,
    _ffun,
    _anfun,
    _bnfun,
    _dK_dp,
    _dK_dq,
    _dan_dp,
    _dan_dq,
    _dbn_dp,
    _dbn_dq,
    _dnextapp,
)

# Default constants for the continued fraction approximation
# These can be overridden via keyword arguments to betainc()
DEFAULT_EPSILON = 1e-14  # Tighter convergence for better accuracy at extreme values
DEFAULT_MIN_APPROX = 3
DEFAULT_MAX_APPROX = 500  # Increased iterations for better convergence near boundaries


class Betainc(Function):
    """
    Custom autograd function for the regularized incomplete beta function.
    
    This implements I_x(a, b), the regularized incomplete beta function,
    with full gradient support for all three parameters.
    """

    @staticmethod
    def forward(ctx, a, b, x, epsilon, min_approx, max_approx):
        """
        Compute the regularized incomplete beta function I_x(a, b).
        
        Args:
            ctx: Context object for saving tensors for backward pass
            a: First shape parameter (must be positive)
            b: Second shape parameter (must be positive)
            x: Upper limit of integration (must be in [0, 1])
            epsilon: Convergence threshold
            min_approx: Minimum number of iterations
            max_approx: Maximum number of iterations
            
        Returns:
            The value of I_x(a, b)
        """
        ctx.save_for_backward(a, b, x)
        ctx.epsilon = epsilon
        ctx.min_approx = min_approx
        ctx.max_approx = max_approx
        
        # Handle broadcasting
        final_shape = torch.broadcast_shapes(a.shape, b.shape, x.shape)
        a = a.expand(final_shape)
        b = b.expand(final_shape)
        x = x.expand(final_shape)

        # Create masks for edge cases
        x_le_0 = x <= 0.0
        x_ge_1 = x >= 1.0
        x_intermediate = ~x_le_0 & ~x_ge_1

        # Initialize result tensor
        final_I = torch.zeros_like(x)
        final_I[x_ge_1] = 1.0
        
        # Only compute for values strictly between 0 and 1
        if x_intermediate.any():
            a_comp = a[x_intermediate]
            b_comp = b[x_intermediate]
            x_comp = x[x_intermediate]

            # Use symmetry relation to improve numerical stability
            swapped = x_comp > a_comp / (a_comp + b_comp)
            
            p = torch.where(swapped, b_comp, a_comp)
            q = torch.where(swapped, a_comp, b_comp)
            x_calc = torch.where(swapped, 1.0 - x_comp, x_comp)

            K = _Kfun(p, q, x_calc)
            f = _ffun(p, q, x_calc)

            # Initialize state variables for continued fraction
            one = torch.ones_like(p)
            zero = torch.zeros_like(p)
            App, Ap, Bpp, Bp = one.clone(), one.clone(), zero.clone(), one.clone()
            Ixpq = torch.full_like(p, torch.nan)
            
            # Track convergence
            converged_mask = torch.zeros_like(p, dtype=torch.bool)
            
            # Continued fraction iteration
            for n in range(1, max_approx + 1):
                if converged_mask.all():
                    break

                an, bn = _anfun(p, q, f, n), _bnfun(p, q, f, n)
                An, Bn = an * App + bn * Ap, an * Bpp + bn * Bp
                
                # Prevent division by zero
                Bn = torch.where(torch.abs(Bn) < 1e-30, torch.full_like(Bn, 1e-30), Bn)

                Cn = An / Bn
                Ixpqn = K * Cn
                
                # Update convergence mask
                not_converged = ~converged_mask
                newly_converged = (torch.abs(Ixpqn - Ixpq) < epsilon) & not_converged & (n >= min_approx)
                converged_mask[newly_converged] = True
                
                # Update values for non-converged elements
                Ixpq = torch.where(not_converged, Ixpqn, Ixpq)
                
                # Update state variables
                App, Ap = torch.where(not_converged.unsqueeze(0), torch.stack([Ap, An]), torch.stack([App, Ap]))
                Bpp, Bp = torch.where(not_converged.unsqueeze(0), torch.stack([Bp, Bn]), torch.stack([Bpp, Bp]))

            # Apply symmetry relation if we swapped
            intermediate_result = torch.where(swapped, 1.0 - Ixpq, Ixpq)
            final_I[x_intermediate] = intermediate_result

        return final_I

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradients of the incomplete beta function.
        
        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient of the loss with respect to the output
            
        Returns:
            Tuple of gradients (grad_a, grad_b, grad_x)
        """
        a, b, x = ctx.saved_tensors
        epsilon = ctx.epsilon
        min_approx = ctx.min_approx
        max_approx = ctx.max_approx
        grad_a = grad_b = grad_x = None

        final_shape = torch.broadcast_shapes(a.shape, b.shape, x.shape)
        a = a.expand(final_shape)
        b = b.expand(final_shape)
        x = x.expand(final_shape)

        x_le_0 = x <= 0.0
        x_ge_1 = x >= 1.0
        x_intermediate = ~x_le_0 & ~x_ge_1

        # Initialize gradients only if needed
        if ctx.needs_input_grad[0]:
            grad_a = torch.zeros_like(a)
        if ctx.needs_input_grad[1]:
            grad_b = torch.zeros_like(b)
        if ctx.needs_input_grad[2]:
            grad_x = torch.zeros_like(x)

        if x_intermediate.any():
            a_comp = a[x_intermediate]
            b_comp = b[x_intermediate]
            x_comp = x[x_intermediate]

            swapped = x_comp > a_comp / (a_comp + b_comp)
            
            p = torch.where(swapped, b_comp, a_comp)
            q = torch.where(swapped, a_comp, b_comp)
            x_calc = torch.where(swapped, 1.0 - x_comp, x_comp)

            K = _Kfun(p, q, x_calc)
            f = _ffun(p, q, x_calc)
            psi_p, psi_q, psi_pq = (
                torch.special.digamma(p),
                torch.special.digamma(q),
                torch.special.digamma(p + q),
            )
            dK_d_p = _dK_dp(x_calc, p, q, K, psi_pq, psi_p)
            dK_d_q = _dK_dq(x_calc, p, q, K, psi_pq, psi_q)

            one = torch.ones_like(p)
            zero = torch.zeros_like(p)
            App, Ap, Bpp, Bp = one.clone(), one.clone(), zero.clone(), one.clone()
            dApp_dp, dAp_dp, dBpp_dp, dBp_dp = [z.clone() for z in [zero, zero, zero, zero]]
            dApp_dq, dAp_dq, dBpp_dq, dBp_dq = [z.clone() for z in [zero, zero, zero, zero]]
            
            Ixpq = torch.full_like(p, torch.nan)
            dI_dp = torch.zeros_like(p)
            dI_dq = torch.zeros_like(p)
            
            converged_mask = torch.zeros_like(p, dtype=torch.bool)
            
            for n in range(1, max_approx + 1):
                if converged_mask.all():
                    break
                    
                an, bn = _anfun(p, q, f, n), _bnfun(p, q, f, n)
                An, Bn = an * App + bn * Ap, an * Bpp + bn * Bp
                
                dan_p, dbn_p = _dan_dp(p, q, f, n), _dbn_dp(p, q, f, n)
                dAn_dp = _dnextapp(an, bn, dan_p, dbn_p, App, Ap, dApp_dp, dAp_dp)
                dBn_dp = _dnextapp(an, bn, dan_p, dbn_p, Bpp, Bp, dBpp_dp, dBp_dp)
                
                dan_q, dbn_q = _dan_dq(p, q, f, n), _dbn_dq(p, q, f, n)
                dAn_dq = _dnextapp(an, bn, dan_q, dbn_q, App, Ap, dApp_dq, dAp_dq)
                dBn_dq = _dnextapp(an, bn, dan_q, dbn_q, Bpp, Bp, dBpp_dq, dBp_dq)

                Bn_safe = torch.where(torch.abs(Bn) < 1e-30, torch.full_like(Bn, 1e-30), Bn)
                
                Cn = An / Bn_safe
                Ixpqn = K * Cn
                
                current_dI_dp = dK_d_p * Cn + K * ((dAn_dp / Bn_safe) - (An * dBn_dp / Bn_safe.pow(2)))
                current_dI_dq = dK_d_q * Cn + K * ((dAn_dq / Bn_safe) - (An * dBn_dq / Bn_safe.pow(2)))
                
                not_converged = ~converged_mask
                newly_converged = (torch.abs(Ixpqn - Ixpq) < epsilon) & not_converged & (n >= min_approx)
                converged_mask[newly_converged] = True
                
                Ixpq = torch.where(not_converged, Ixpqn, Ixpq)
                dI_dp = torch.where(not_converged, current_dI_dp, dI_dp)
                dI_dq = torch.where(not_converged, current_dI_dq, dI_dq)
                
                App_s, Ap_s = torch.stack([App, Ap])
                Bpp_s, Bp_s = torch.stack([Bpp, Bp])
                dApp_dp_s, dAp_dp_s = torch.stack([dApp_dp, dAp_dp])
                dBpp_dp_s, dBp_dp_s = torch.stack([dBpp_dp, dBp_dp])
                dApp_dq_s, dAp_dq_s = torch.stack([dApp_dq, dAp_dq])
                dBpp_dq_s, dBp_dq_s = torch.stack([dBpp_dq, dBp_dq])
                
                App, Ap = torch.where(not_converged, torch.stack([Ap, An]), App_s)
                Bpp, Bp = torch.where(not_converged, torch.stack([Bp, Bn]), Bpp_s)
                dApp_dp, dAp_dp = torch.where(not_converged, torch.stack([dAp_dp, dAn_dp]), dApp_dp_s)
                dBpp_dp, dBp_dp = torch.where(not_converged, torch.stack([dBp_dp, dBn_dp]), dBpp_dp_s)
                dApp_dq, dAp_dq = torch.where(not_converged, torch.stack([dAp_dq, dAn_dq]), dApp_dq_s)
                dBpp_dq, dBp_dq = torch.where(not_converged, torch.stack([dBp_dq, dBn_dq]), dBpp_dq_s)

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_a_unscaled = torch.where(swapped, -dI_dq, dI_dp)
                grad_b_unscaled = torch.where(swapped, -dI_dp, dI_dq)
                
                if ctx.needs_input_grad[0]:
                    grad_a[x_intermediate] = grad_a_unscaled
                if ctx.needs_input_grad[1]:
                    grad_b[x_intermediate] = grad_b_unscaled

        if ctx.needs_input_grad[2]:
            grad_x_unscaled = x.pow(a - 1.0) * (1.0 - x).pow(b - 1.0) / _beta(a, b)
            grad_x[x_intermediate] = grad_x_unscaled[x_intermediate]
            
        # Apply chain rule
        grad_a = grad_a * grad_output if ctx.needs_input_grad[0] else None
        grad_b = grad_b * grad_output if ctx.needs_input_grad[1] else None
        grad_x = grad_x * grad_output if ctx.needs_input_grad[2] else None
            
        # Return None for epsilon, min_approx, max_approx (not differentiable)
        return grad_a, grad_b, grad_x, None, None, None


def betainc(a, b, x, epsilon=DEFAULT_EPSILON, min_approx=DEFAULT_MIN_APPROX, max_approx=DEFAULT_MAX_APPROX):
    """
    Compute the regularized incomplete beta function I_x(a, b).
    
    The regularized incomplete beta function is defined as:
    
        I_x(a, b) = B(x; a, b) / B(a, b)
    
    where B(x; a, b) is the incomplete beta function and B(a, b) is the 
    complete beta function.
    
    This implementation is fully differentiable with respect to all three
    parameters and supports batched computation with tensor inputs.
    
    Args:
        a (torch.Tensor): First shape parameter. Must be positive.
        b (torch.Tensor): Second shape parameter. Must be positive.
        x (torch.Tensor): Upper limit of integration. Must be in [0, 1].
        epsilon (float, optional): Convergence threshold for continued fraction.
            Default: 1e-14 (tighter for better accuracy at extreme values).
        min_approx (int, optional): Minimum number of iterations before checking
            convergence. Default: 3.
        max_approx (int, optional): Maximum number of iterations for continued
            fraction approximation. Default: 500 (increased for better convergence
            near boundaries).
        
    Returns:
        torch.Tensor: The value of I_x(a, b). The output shape is determined
            by broadcasting the input shapes.
            
    Examples:
        >>> import torch
        >>> from torch_betainc import betainc
        >>> 
        >>> # Single values
        >>> a = torch.tensor(2.0, requires_grad=True)
        >>> b = torch.tensor(3.0, requires_grad=True)
        >>> x = torch.tensor(0.5, requires_grad=True)
        >>> result = betainc(a, b, x)
        >>> print(result)
        tensor(0.6875, grad_fn=<BetaincBackward>)
        >>> 
        >>> # Batch computation
        >>> a = torch.tensor([1.0, 2.0, 3.0])
        >>> b = torch.tensor([1.0, 2.0, 3.0])
        >>> x = torch.tensor([0.3, 0.5, 0.7])
        >>> result = betainc(a, b, x)
        >>> print(result)
        tensor([0.3000, 0.5000, 0.7840])
        >>> 
        >>> # Gradient computation
        >>> a = torch.tensor(2.0, requires_grad=True)
        >>> b = torch.tensor(3.0, requires_grad=True)
        >>> x = torch.tensor(0.5, requires_grad=True)
        >>> result = betainc(a, b, x)
        >>> result.backward()
        >>> print(f"∂I/∂a = {a.grad}")
        >>> print(f"∂I/∂b = {b.grad}")
        >>> print(f"∂I/∂x = {x.grad}")
        >>> 
        >>> # Custom precision settings
        >>> # Use original precision for faster computation
        >>> result = betainc(a, b, x, epsilon=1e-12, max_approx=200)
        
    References:
        Based on the implementation by Arthur Zwaenepoel:
        https://github.com/arzwa/IncBetaDer
    """
    return Betainc.apply(a, b, x, epsilon, min_approx, max_approx)
