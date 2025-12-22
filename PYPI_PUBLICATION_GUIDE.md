# PyPI Publication Guide for torch-betainc

## Pre-Publication Checklist

✅ **Completed:**
- [x] Enhanced LICENSE with Arthur Zwaenepoel attribution
- [x] Updated pyproject.toml with correct GitHub URLs
- [x] Added link to original work in project URLs
- [x] Fixed README Performance Considerations section
- [x] Created CHANGELOG.md
- [x] Built package successfully

📦 **Build artifacts created:**
- `dist/torch_betainc-0.1.0.tar.gz` (source distribution)
- `dist/torch_betainc-0.1.0-py3-none-any.whl` (wheel)

## Publication Steps

### Step 1: Test Installation Locally

Test the built package in a clean environment:

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate

# Install from wheel
pip install dist/torch_betainc-0.1.0-py3-none-any.whl

# Test import
python -c "from torch_betainc import betainc, cdf_t; print('✓ Import successful')"

# Run a quick test
python -c "
import torch
from torch_betainc import betainc
result = betainc(torch.tensor(2.0), torch.tensor(3.0), torch.tensor(0.5))
print(f'betainc(2, 3, 0.5) = {result.item():.4f}')
assert abs(result.item() - 0.6875) < 0.001, 'Test failed!'
print('✓ Basic test passed')
"

# Clean up
deactivate
rm -rf test_env
```

### Step 2: Publish to TestPyPI (Optional but Recommended)

TestPyPI is a separate instance of PyPI for testing purposes.

```bash
# Activate your environment
source .venv-betainc/bin/activate

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# You'll be prompted for:
# - Username: __token__
# - Password: your TestPyPI API token
```

**Get TestPyPI token:**
1. Go to https://test.pypi.org/manage/account/
2. Scroll to "API tokens"
3. Click "Add API token"
4. Name it (e.g., "torch-betainc")
5. Copy the token (starts with `pypi-`)

**Test installation from TestPyPI:**

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ torch-betainc
```

Note: `--extra-index-url` is needed because dependencies (like PyTorch) are on the main PyPI.

### Step 3: Publish to PyPI

Once you've verified everything works:

```bash
# Activate your environment
source .venv-betainc/bin/activate

# Upload to PyPI
python -m twine upload dist/*

# You'll be prompted for:
# - Username: __token__
# - Password: your PyPI API token
```

**Get PyPI token:**
1. Go to https://pypi.org/manage/account/
2. Scroll to "API tokens"
3. Click "Add API token"
4. Scope: "Entire account" or specific to this project
5. Name it (e.g., "torch-betainc")
6. Copy the token (starts with `pypi-`)

**⚠️ Important:** Save your API token securely! You won't be able to see it again.

### Step 4: Verify Publication

After publishing:

```bash
# Wait a few minutes, then install from PyPI
pip install torch-betainc

# Test it works
python -c "from torch_betainc import betainc; print('✓ Installed from PyPI successfully')"
```

Check your package page: https://pypi.org/project/torch-betainc/

### Step 5: Create GitHub Release

1. Go to https://github.com/k-onoue/torch-betainc/releases
2. Click "Create a new release"
3. Tag: `v0.1.0`
4. Title: `v0.1.0 - Initial Release`
5. Description: Copy from CHANGELOG.md
6. Attach the built files from `dist/`
7. Publish release

## Post-Publication

### Update README Badge (Optional)

Add PyPI badges to your README:

```markdown
[![PyPI version](https://badge.fury.io/py/torch-betainc.svg)](https://badge.fury.io/py/torch-betainc)
[![Downloads](https://pepy.tech/badge/torch-betainc)](https://pepy.tech/project/torch-betainc)
```

### Contact Arthur Zwaenepoel (Optional but Recommended)

Send a brief, friendly email:

```
Subject: torch-betainc: PyTorch Package Based on Your IncBetaDer Work

Hi Arthur,

I wanted to let you know that I've created a PyTorch package called 
torch-betainc, which is based on your excellent IncBetaDer implementation.

The package extends your work with:
- Full PyTorch integration with autograd support
- Vectorization for batch processing
- Student's t-distribution CDF
- Comprehensive tests and documentation

I've published it to PyPI: https://pypi.org/project/torch-betainc/
GitHub: https://github.com/k-onoue/torch-betainc

Your work is prominently credited in the README, LICENSE, and source code.
I hope this makes differentiable beta functions more accessible to the 
PyTorch community!

Thank you for your original implementation.

Best regards,
Keisuke Onoue
```

## Troubleshooting

### Build Warnings

The deprecation warnings about `project.license` format are non-critical. They can be fixed in a future version by updating to the newer SPDX format in pyproject.toml.

### Upload Errors

If you get "File already exists" error:
- You can't re-upload the same version
- Increment version number in pyproject.toml
- Rebuild: `python -m build`
- Upload again

### Installation Issues

If users report installation issues:
- Check PyTorch compatibility
- Ensure dependencies are correctly specified
- Test in different Python versions (3.8-3.12)

## Future Versions

When releasing updates:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Update `__version__` in `torch_betainc/__init__.py`
4. Rebuild: `python -m build`
5. Upload: `python -m twine upload dist/*`
6. Create GitHub release

## Resources

- PyPI: https://pypi.org/
- TestPyPI: https://test.pypi.org/
- Packaging Guide: https://packaging.python.org/
- Twine docs: https://twine.readthedocs.io/
