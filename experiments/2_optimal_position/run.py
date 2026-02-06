"""
实验 2：最优位置与一致性度量 (Optimal Position pos* & Consistency Metric)
对应论文：Section 4.3, Appendix (Metric for Maximum Benefit Position, Enhanced Results Analysis)
输出：output/optimal/
"""
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_exp = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, _exp)
os.chdir(_root)

from optimal_position_verification import main
main()
