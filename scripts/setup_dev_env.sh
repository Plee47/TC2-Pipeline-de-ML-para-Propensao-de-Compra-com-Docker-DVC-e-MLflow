#!/bin/bash
set -e

echo "Installing project dependencies..."

# Install Poetry if not available
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    python -m pip install poetry --quiet
fi

# Install project dependencies
poetry install

echo "Setup complete!"
