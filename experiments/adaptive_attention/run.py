"""Run adaptive attention experiment. Execute from repo root: python experiments/adaptive_attention/run.py"""
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.chdir(_root)

from adaptive_attention_mechanism import main
main()
