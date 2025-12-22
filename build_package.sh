#!/bin/bash
# Build and test the package before publishing to PyPI

set -e  # Exit on error

echo "=========================================="
echo "Building torch-betainc package"
echo "=========================================="

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
echo "   pip install dist/torch_betainc-0.1.0-py3-none-any.whl"
echo "   python -c 'from torch_betainc import betainc; print(betainc)'"
echo ""
echo "2. Publish to TestPyPI (optional):"
echo "   python -m twine upload --repository testpypi dist/*"
echo ""
echo "3. Publish to PyPI:"
echo "   python -m twine upload dist/*"
echo ""
