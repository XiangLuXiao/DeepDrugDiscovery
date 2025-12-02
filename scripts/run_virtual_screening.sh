#!/usr/bin/env bash
set -euo pipefail

# Always run from repository root so Python can resolve local packages
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Compute similarity metrics
echo
echo ">>> Compute similarity metrics on generated representations"
python -m experiments.generate_similarity_metrics "$@"

# Ligand-based virtual screening
echo
echo ">>> Perform ligand-based virtual screening"
python -m experiments.ligand_based_virtual_screening "$@"
