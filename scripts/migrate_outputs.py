"""One-time script: copy existing *_output and triple_attention_plots into output/. Run from repo root: python scripts/migrate_outputs.py"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MAPPING = [
    ("position_output", "output/position"),
    ("optimal_output", "output/optimal"),
    ("parameter_output", "output/parameter"),
    ("concrete_output", "output/concrete"),
    ("consistency_output", "output/consistency"),
    ("adaptive_output", "output/adaptive"),
    ("triple_attention_plots", "output/triple_attention"),
]
for src, dst in MAPPING:
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(src):
            s, d = os.path.join(src, name), os.path.join(dst, name)
            if os.path.isfile(s):
                shutil.copy2(s, d)
            else:
                shutil.copytree(s, d, dirs_exist_ok=True)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"Skip (not found): {src}")
print("Done.")
