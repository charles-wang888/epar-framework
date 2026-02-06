"""
Run all experiment categories in order. Outputs go to output/<category>/.
Execute from repo root: python run_all.py
"""
import sys
import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# 按论文实验编号调用各 experiments/<n_xxx>/run.py；(script, 额外参数列表)
RUNNERS = [
    ("1_position_effect", "experiments/1_position_effect/run.py", []),
    ("2_optimal_position", "experiments/2_optimal_position/run.py", []),
    ("3_parameter_sensitivity", "experiments/3_parameter_sensitivity/run.py", []),
    ("4_adaptive_attention", "experiments/4_adaptive_attention/run.py", [])
]

def main():
    print("=== EPAR: Running all experiment categories ===\n")
    for name, script, extra in RUNNERS:
        print(f"\n>>> {name}")
        cmd = [sys.executable, script] + extra
        print(" ".join(cmd))
        ret = subprocess.run(cmd, cwd=ROOT)
        if ret.returncode != 0:
            print(f"Warning: {name} exited with code {ret.returncode}")
    print("\n=== All runs finished. Outputs are under output/ ===\n")

if __name__ == "__main__":
    main()
