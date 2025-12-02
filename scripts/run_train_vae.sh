#!/usr/bin/env bash
set -euo pipefail

# Always run from repository root so Python can resolve local packages
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Train baseline representation model
echo
echo ">>> Train baseline molecular VAE representation"
python -m experiments.train_representation "$@"

# Generate latent representations
echo
echo ">>> Generate latent representations for the dataset"
python -m experiments.generate_representation "$@"
