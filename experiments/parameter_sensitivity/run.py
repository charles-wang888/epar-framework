"""Run parameter sensitivity analysis and improvement chart. Execute from repo root: python experiments/parameter_sensitivity/run.py"""
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.chdir(_root)

from parameter_sensitivity_analysis import main
main()
print("\n--- Creating improvement chart ---")
from create_improvement_chart import create_improvement_chart
create_improvement_chart()
