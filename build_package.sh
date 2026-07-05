#!/bin/bash
# Build and test the package before publishing to PyPI

set -e  # Exit on error

echo "=========================================="
echo "Building torch-betainc package"
echo "=========================================="

VERSION=$(grep -m1 '^version = ' pyproject.toml | cut -d '"' -f2)

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info torch_betainc.egg-info

# Build the package
echo "Building package..."
python -m build

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "Generated files:"
ls -lh dist/

echo ""
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo ""
echo "1. Test installation in clean environment:"
echo "   python -m venv test_env"
echo "   source test_env/bin/activate"
echo "   pip install dist/torch_betainc-*.whl"
echo "   python -c 'from torch_betainc import betainc, StudentT; print(betainc, StudentT)'"
echo ""
echo "2. Publish to PyPI via GitHub Actions:"
echo "   git tag v${VERSION}"
echo "   git push origin v${VERSION}"
echo ""
