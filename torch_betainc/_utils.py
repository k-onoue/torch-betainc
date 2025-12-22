"""
Internal helper functions for the incomplete beta function computation.

These functions are not part of the public API and are prefixed with underscore
to indicate they are private implementation details.
"""

import torch


def _beta(p, q):
    """Compute the beta function B(p, q) using log-gamma."""
    log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
    return torch.exp(log_beta)


def _Kfun(p, q, x_calc):
    """Compute the K function used in the continued fraction expansion."""
    return (x_calc.pow(p) * (1.0 - x_calc).pow(q - 1)) / (p * _beta(p, q))


def _ffun(p, q, x_calc):
    """Compute the f function used in the continued fraction expansion."""
    return q * x_calc / (p * (1.0 - x_calc))


def _a1fun(p, q, f):
    """Compute the first coefficient a_1 in the continued fraction."""
    return p * f * (q - 1.0) / (q * (p + 1.0))


def _anfun(p, q, f, n):
    """Compute the n-th coefficient a_n in the continued fraction."""
    if n == 1:
        return _a1fun(p, q, f)
    return (
        p.pow(2)
        * f.pow(2)
        * (n - 1.0)
        * (p + q + n - 2.0)
        * (p + n - 1.0)
        * (q - n)
        / (q.pow(2) * (p + 2 * n - 3.0) * (p + 2 * n - 2.0).pow(2) * (p + 2 * n - 1.0))
    )


def _bnfun(p, q, f, n):
    """Compute the n-th coefficient b_n in the continued fraction."""
    x_ = (
        2 * (p * f + 2 * q) * n**2
        + 2 * (p * f + 2 * q) * (p - 1.0) * n
        + p * q * (p - 2.0 - p * f)
    )
    y_ = q * (p + 2 * n - 2.0) * (p + 2 * n)
    return x_ / y_


def _dK_dp(x, p, q, K, psi_pq, psi_p):
    """Compute the derivative of K with respect to p."""
    return K * (torch.log(x) - 1.0 / p + psi_pq - psi_p)


def _dK_dq(x, p, q, K, psi_pq, psi_q):
    """Compute the derivative of K with respect to q."""
    return K * (torch.log(1.0 - x) + psi_pq - psi_q)


def _da1_dp(p, q, f):
    """Compute the derivative of a_1 with respect to p."""
    return -p * f * (q - 1.0) / (q * (p + 1.0).pow(2))


def _dan_dp(p, q, f, n):
    """Compute the derivative of a_n with respect to p."""
    if n == 1:
        return _da1_dp(p, q, f)
    x = -(n - 1.0) * f.pow(2) * p.pow(2) * (q - n)
    y = (
        (-1.0 + p + q) * 8 * n**3
        + (16 * p.pow(2) + (-44.0 + 20 * q) * p + 26.0 - 24 * q) * n**2
        + (
            10 * p.pow(3)
            + (14 * q - 46.0) * p.pow(2)
            + (-40 * q + 66.0) * p
            - 28.0
            + 24 * q
        )
        * n
        + 2 * p.pow(4)
        + (-13 + 3 * q) * p.pow(3)
        + (16 - 14 * q) * p.pow(2)
        + (-29 + 19 * q) * p
        + 10.0
        - 8 * q
    )
    z = (
        q.pow(2)
        * (p + 2 * n - 3.0).pow(2)
        * (p + 2 * n - 2.0).pow(3)
        * (p + 2 * n - 1.0).pow(2)
    )
    return x * y / z


def _da1_dq(p, q, f):
    """Compute the derivative of a_1 with respect to q."""
    return f * p / (q * (p + 1.0))


def _dan_dq(p, q, f, n):
    """Compute the derivative of a_n with respect to q."""
    if n == 1:
        return _da1_dq(p, q, f)
    x = p.pow(2) * f.pow(2) * (n - 1.0) * (p + n - 1.0) * (2 * q + p - 2.0)
    y = q.pow(2) * (p + 2 * n - 3.0) * (p + 2 * n - 2.0).pow(2) * (p + 2 * n - 1.0)
    return x / y


def _dbn_dp(p, q, f, n):
    """Compute the derivative of b_n with respect to p."""
    x = (
        (1.0 - p - q) * 4 * n**2
        + (4 * p - 4.0 + 4 * q - 2 * p.pow(2)) * n
        + p.pow(2) * q
    )
    y = q * (p + 2 * n - 2.0).pow(2) * (p + 2 * n).pow(2)
    return p * f * x / y


def _dbn_dq(p, q, f, n):
    """Compute the derivative of b_n with respect to q."""
    return -p.pow(2) * f / (q * (p + 2 * n - 2.0) * (p + 2 * n))


def _dnextapp(an, bn, dan, dbn, Xpp, Xp, dXpp, dXp):
    """Compute the next approximation in the continued fraction."""
    return dan * Xpp + an * dXpp + dbn * Xp + bn * dXp
