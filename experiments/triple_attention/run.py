"""Run triple-attention experiments. Execute from repo root: python experiments/triple_attention/run.py"""
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.chdir(_root)

from run_triple_attention_experiments import main
main()
