#!/usr/bin/env bash
set -euo pipefail

# Always run from repository root so Python can resolve local packages
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# # Evaluate reconstruction quality
# echo
# echo ">>> Evaluate reconstruction quality of the VAE"
# python -m experiments.evaludate_reconstruction "$@"

# # Diversity analyses
# echo
# echo ">>> Analyze molecular diversity in the dataset"
# python -m experiments.diversity_analysis "$@"

echo
echo ">>> Compute compound-level diversity statistics"
python -m experiments.compound_diversity "$@"

# # Leave-one-out evaluation
# echo
# echo ">>> Run leave-one-out evaluation"
# python -m experiments.leave_one_out "$@"
