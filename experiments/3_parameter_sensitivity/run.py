"""
实验 3：参数敏感性 (α, β, γ) 与系统消融
对应论文：Appendix (Systematic Ablation Study, Gamma Parameter Optimization, Parameter Selection Theory)
输出：output/parameter/
"""
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_exp = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, _exp)
os.chdir(_root)

from parameter_sensitivity_analysis import main
main()
print("\n--- Creating improvement chart ---")
from create_improvement_chart import create_improvement_chart
create_improvement_chart()
