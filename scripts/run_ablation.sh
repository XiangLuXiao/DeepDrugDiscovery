#!/usr/bin/env bash
set -euo pipefail

# Always run from repository root so Python can resolve local packages
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# # Ablation: training variants
# echo
# echo ">>> Ablation: train with GRU addition variant"
# python -m experiments.ablation.train_representation_ablation_gru_add "$@"

# echo
# echo ">>> Ablation: train with GRU concatenation variant"
# python -m experiments.ablation.train_representation_ablation_gru_concat "$@"

# echo
# echo ">>> Ablation: train with selected features removed"
# python -m experiments.ablation.train_representation_ablation_remove_features "$@"

# Ablation: representation generation
echo
echo ">>> Ablation: generate representations (GRU addition variant)"
python -m experiments.ablation.generate_representation_ablation_gru_add "$@"

echo
echo ">>> Ablation: generate representations (GRU concatenation variant)"
python -m experiments.ablation.generate_representation_ablation_gru_concat "$@"

echo
echo ">>> Ablation: generate representations without removed features"
python -m experiments.ablation.generate_representation_ablation_remove_features "$@"

# Ablation: downstream evaluation
echo
echo ">>> Ablation: compute similarity metrics on ablation outputs"
python -m experiments.ablation.generate_similarity_metrics_ablation "$@"

echo
echo ">>> Ablation: ligand-based virtual screening with ablated models"
python -m experiments.ablation.ligand_based_virtual_screening_ablation "$@"
