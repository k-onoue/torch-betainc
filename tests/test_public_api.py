"""
Tests for the package-level public API.
"""

import torch_betainc
from torch_betainc import StudentT, betainc


def test_public_api_exports():
    """Only the documented public API should be exported at package level."""
    assert torch_betainc.__all__ == ["betainc", "StudentT"]
    assert torch_betainc.betainc is betainc
    assert torch_betainc.StudentT is StudentT
    assert not hasattr(torch_betainc, "cdf_t")
