#!/bin/bash

# NiChart_SPARE Installation Script
# This script installs the NiChart_SPARE package

set -e  # Exit on any error

echo "=========================================="
echo "NiChart_SPARE Installation Script"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.7"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python version $PYTHON_VERSION is too old"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

echo "Python version: $PYTHON_VERSION ✓"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed or not in PATH"
    echo "Please install pip3"
    exit 1
fi

echo "pip3 found ✓"

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Install the package in development mode
echo "Installing NiChart_SPARE package..."
pip3 install -e .

echo ""
echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo ""
echo "You can now use NiChart_SPARE from the command line:"
echo ""
echo "  # Training example:"
echo "  NiChart_SPARE -a trainer -t AD -i input.csv -mo model.pkl.gz -v True"
echo ""
echo "  # Inference example:"
echo "  NiChart_SPARE -a inference -t AD -i test.csv -mo model.pkl.gz -v False -o output.csv"
echo ""
echo "  # Get help:"
echo "  NiChart_SPARE --help"
echo ""
echo "For more information, visit: https://github.com/CBICA/NiChart_SPARE" 