"""
EPAR: Position-Aware Attention core package.
Ensure project root is on path so config can be imported from any cwd.
"""
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from epar.position_attention_model import (
    PositionAttentionConfig,
    PositionAwareAttention,
    PositionEffectFunction,
    PositionAttentionExperiment,
    main as run_position_experiment,
)

__all__ = [
    "PositionAttentionConfig",
    "PositionAwareAttention",
    "PositionEffectFunction",
    "PositionAttentionExperiment",
    "run_position_experiment",
]
