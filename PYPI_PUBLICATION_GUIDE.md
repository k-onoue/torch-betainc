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
- `dist/torch_betainc-<version>.tar.gz` (source distribution)
- `dist/torch_betainc-<version>-py3-none-any.whl` (wheel)

## Publication Steps

### Step 1: Configure PyPI Trusted Publishing

This repository publishes to PyPI automatically from `.github/workflows/publish.yml`
when a `v*` tag is pushed. The workflow uses PyPI Trusted Publishing, so no
PyPI token or GitHub secret is required.

Configure this once in PyPI:

1. Go to https://pypi.org/manage/account/publishing/
2. Add a publisher for project `torch-betainc`
3. Use GitHub owner `k-onoue`
4. Use repository name `torch-betainc`
5. Use workflow name `publish.yml`
6. Use environment name `pypi`

### Step 2: Test Installation Locally

Test the built package in a clean environment:

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate

# Install from wheel
pip install dist/torch_betainc-*.whl

# Test import
python -c "from torch_betainc import betainc, StudentT; print('✓ Import successful')"

# Run a quick test
python -c "
import torch
from torch_betainc import betainc, StudentT
result = betainc(torch.tensor(2.0), torch.tensor(3.0), torch.tensor(0.5))
print(f'betainc(2, 3, 0.5) = {result.item():.4f}')
assert abs(result.item() - 0.6875) < 0.001, 'Test failed!'
dist = StudentT(df=torch.tensor(5.0))
cdf = dist.cdf(torch.tensor(1.0))
assert 0.0 < cdf.item() < 1.0, 'StudentT CDF test failed!'
print('✓ Basic test passed')
"

# Clean up
deactivate
rm -rf test_env
```

### Step 3: Publish to PyPI Automatically

Once the version has been updated and the release commit is on `main`, push a
release tag:

```bash
git tag v0.2.1
git push origin v0.2.1
```

The GitHub Actions workflow will then:

- run the test suite
- build the source distribution and wheel
- publish the artifacts to PyPI

PyPI versions are immutable, so make sure `pyproject.toml` and
`torch_betainc/__init__.py` contain a version that has not been uploaded before.

### Step 4: Verify Publication

After the workflow succeeds:

```bash
pip install torch-betainc

python -c "from torch_betainc import betainc, StudentT; print('✓ Installed from PyPI successfully')"
```

Check your package page: https://pypi.org/project/torch-betainc/

### Step 5: Create GitHub Release

1. Go to https://github.com/k-onoue/torch-betainc/releases
2. Click "Create a new release"
3. Tag: `v0.2.1`
4. Title: `v0.2.1`
5. Description: Copy from CHANGELOG.md
6. Publish release

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
- Update `torch_betainc/__init__.py`
- Create and push a new release tag

### Trusted Publishing Errors

If the workflow fails with a trusted publisher error, check the PyPI publisher
settings. The project name, GitHub owner, repository, workflow filename, and
environment name must match the workflow exactly.

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
4. Commit and push the changes to `main`
5. Create and push a `v*` tag, for example `git tag v0.2.1 && git push origin v0.2.1`
6. Create GitHub release

## Resources

- PyPI: https://pypi.org/
- TestPyPI: https://test.pypi.org/
- Packaging Guide: https://packaging.python.org/
- PyPI Trusted Publishing: https://docs.pypi.org/trusted-publishers/
