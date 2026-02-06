# 实验与论文章节对应关系

本文档说明代码库中每组实验与论文正文/附录的对应关系，便于复现与审稿对照。

## 实验目录与论文对应表

| 实验目录 | 论文位置 | 内容摘要 | 输出目录 |
|----------|----------|----------|----------|
| **1_position_effect** | **Section 4** Position Effect Function；Appendix Mathematical Characteristics, Attention Result Diagram | 位置效应函数 \(P_{\text{effect}}(i,j,L)\)、数学性质（连续性、可微、单调）、注意力统计与热力图 | `output/position/` |
| **2_optimal_position** | **Section 4.3** Information Importance & Optimal Position；Appendix Metric for Maximum Benefit Position, Enhanced Results Analysis | 信息重要性 \(I_j\)、最优位置 \(\text{pos}^* = \arg\max_i V(i)\)、一致性度量与排序相关、增强公式下各信息分布结果 | `output/optimal/` |
| **3_parameter_sensitivity** | **Appendix** Systematic Ablation Study, Gamma Parameter Optimization, Parameter Selection Theory | \(\alpha,\beta,\gamma\) 敏感性、Basic vs Enhanced、消融与改进曲线 | `output/parameter/` |
| **4_adaptive_attention** | **Section 5.4** When and Why the Method Helps；任务/内容感知分析 | 自适应/任务感知注意力、何时使用 EPAR、参数与任务类型分析 | `output/adaptive/` |
| **5_triple_attention** | Appendix 架构与融合权重分析 | 三重注意力架构对比、融合权重敏感性、消融与效率 | `output/triple_attention/` |
| **Supplementary**（longbench, scrolls, wmt_zh_en, codexglue） | **Section 5** 补充实验；Limitations / Future 中 LongBench, SCROLLS, WMT Zh-En, CodeXGlue | 长上下文基准、长文档、非英/代码等扩展验证 | `experiments/results/` |

## 主文实验 (Section 5) 与代码关系

- **5.1 Experimental Setup**：协议与环境见 `experiments/configs/paper_protocol.json`，各实验脚本默认与该协议一致（如 110M、seeds 42–46）。
- **5.2 Comparison with Baseline Methods**：主表结果来自完整训练流程；本仓库提供位置效应、最优位置、参数敏感性等**分析脚本**与**补充基准占位**（LongBench/SCROLLS/WMT/CodeXGlue 需自行接数据与训练）。
- **5.3 Preliminary Llama-2-7B**：为初步实验，当前仓库以 110M 分析为主。
- **5.4 When and Why**：对应实验 4（adaptive_attention）及实验 2/3 中与一致性、\(\gamma\)、信息分布相关的部分。

## 如何按论文复现

1. **位置效应与数学性质**：`python experiments/1_position_effect/run.py` → 查看 `output/position/`。
2. **最优位置与一致性**：`python experiments/2_optimal_position/run.py` → 查看 `output/optimal/`。
3. **参数与消融**：`python experiments/3_parameter_sensitivity/run.py` → 查看 `output/parameter/`。
4. **何时/为何有效**：`python experiments/4_adaptive_attention/run.py` → 查看 `output/adaptive/`。
5. **三重注意力**：`python experiments/5_triple_attention/run.py` → 查看 `output/triple_attention/`。
6. **补充基准**：在 `experiments/` 下运行 `python run_supplementary_experiments.py --benchmark <longbench|scrolls|wmt_zh_en|codexglue>`，结果在 `experiments/results/`。

所有上述命令均需在**仓库根目录**执行（或确保工作目录为仓库根目录）。
