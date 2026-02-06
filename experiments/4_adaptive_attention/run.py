"""
实验 4：自适应/任务感知注意力 (When and Why the Method Helps)
对应论文：Section 5.4 When and Why the Method Helps, task-aware analysis
输出：output/adaptive/
"""
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_exp = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, _exp)
os.chdir(_root)

from adaptive_attention_mechanism import main
main()
