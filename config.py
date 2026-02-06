"""
Project-wide config: unified output directory.
All experiment outputs go under output/<experiment_name>/.
"""
import os

OUTPUT_BASE = "output"

def output_dir(name: str) -> str:
    """Return path to output subdir for experiment `name`. Creates dir if needed."""
    path = os.path.join(OUTPUT_BASE, name)
    os.makedirs(path, exist_ok=True)
    return path

# Canonical names for each experiment category
POSITION_EFFECT = "position"
OPTIMAL_POSITION = "optimal"
PARAMETER_SENSITIVITY = "parameter"
CONCRETE_EXAMPLE = "concrete"
CONSISTENCY = "consistency"
ADAPTIVE_ATTENTION = "adaptive"
TRIPLE_ATTENTION = "triple_attention"
