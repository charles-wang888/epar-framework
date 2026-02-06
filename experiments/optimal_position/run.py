"""Run optimal-position verification. Execute from repo root: python experiments/optimal_position/run.py"""
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.chdir(_root)

from optimal_position_verification import main
main()
