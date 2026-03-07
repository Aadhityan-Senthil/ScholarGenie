#!/bin/bash

echo "============================================"
echo "  ScholarGenie - One-Click Setup"
echo "============================================"
echo ""

echo "[1/3] Installing Python packages..."
pip install -r requirements_simple.txt

echo ""
echo "[2/3] Creating data directories..."
mkdir -p data/models
mkdir -p data/papers
mkdir -p data/presentations

echo ""
echo "[3/3] Setup complete!"
echo ""
echo "============================================"
echo "  Ready to use ScholarGenie!"
echo "============================================"
echo ""
echo "Run:  python scholargenie.py"
echo ""
