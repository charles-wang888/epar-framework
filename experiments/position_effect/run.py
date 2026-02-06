"""Run position-effect experiment. Execute from repo root: python experiments/position_effect/run.py"""
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.chdir(_root)

from position_attention_model import main
main()
