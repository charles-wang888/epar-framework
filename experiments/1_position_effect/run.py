"""
实验 1：位置效应函数 (Position Effect Function)
对应论文：Section 4, Appendix (Mathematical Characteristics, Attention Result Diagram)
输出：output/position/
"""
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.chdir(_root)

from epar.position_attention_model import main
main()
