"""
Compatibility stub: run and import from epar.position_attention_model.
All outputs go to output/position/ (see config.py).
"""
from epar.position_attention_model import (
    PositionAttentionConfig,
    PositionAwareAttention,
    PositionEffectFunction,
    PositionAttentionExperiment,
    main,
)

__all__ = [
    "PositionAttentionConfig",
    "PositionAwareAttention",
    "PositionEffectFunction",
    "PositionAttentionExperiment",
    "main",
]

if __name__ == "__main__":
    main()
