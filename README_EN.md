# EPAR: Explicit Position-Attention Relationship

**EPAR: Explicit Position-Attention Relationship for Interpretable Long-Range Attention in Transformers**

This codebase implements the position-aware attention mechanism proposed in the paper "EPAR: Explicit Position-Attention Relationship for Interpretable Long-Range Attention in Transformers". The EPAR framework explicitly models how position affects attention strength at the **attention score level**, rather than encoding position into vector representations, thereby achieving analyzability, tunable parameters, and derivable optimal positions.

## 📋 Table of Contents

- [Core Concepts](#core-concepts)
- [Position Effect Function](#position-effect-function)
- [Code Structure](#code-structure)
- [Experiment Description](#experiment-description)
- [Quick Start](#quick-start)
- [Output Description](#output-description)
- [Citation](#citation)

---

## Core Concepts

### EPAR Framework

The core idea of the EPAR (Explicit Position-Attention Relationship) framework is: **explicitly model how position affects attention strength at the attention score level**, rather than implicitly encoding position information through vector representations.

Fundamental differences from existing methods (RoPE, ALiBi, relative position encoding, etc.):

| Method | Operation Level | Mathematical Form | Position Modeling |
|--------|----------------|-------------------|-------------------|
| **RoPE** | Vector representation | $Q'_i = R_{\theta}(i) Q_i$ | Implicit (rotation) |
| **ALiBi** | Attention score | $A_{ij} = Q_i^T K_j + m \cdot \|i-j\|$ | Linear bias |
| **Relative Position Encoding (Shaw)** | Vector representation | $Q'_i = Q_i + P_i$ | Learnable embedding |
| **EPAR (Ours)** | **Attention score** | **$A_{ij} = \text{softmax}((Q_i^T K_j/\sqrt{d_k}) \cdot P_{\text{effect}}(i,j,L))$** | **Explicit function** |

### Main Advantages

1. **Theoretical Guarantees**: Provable optimal parameter selection (Theorem 2) and convergence properties (Theorems 3-5)
2. **Interpretability**: Clear mathematical interpretation of position-attention relationships
3. **Controllability**: Fine-grained control through parameters $\alpha$, $\beta$, $\gamma$
4. **Efficiency**: Position effect matrix can be cached, training overhead increases by only 2.4%

---

## Position Effect Function

### Basic Position Effect Function

The basic position effect function is defined as:

$$P_{\text{effect}}(i,j,L) = \alpha \cdot e^{-\beta \cdot |i-j|/L}$$

Where:
- $\alpha$: Position influence strength parameter (controls intensity)
- $\beta$: Position decay parameter (controls spatial decay rate)
- $L$: Sequence length
- $i, j$: Query position and key position

Based on the position effect function, position-aware attention weights are defined as:

$$A_{ij} = \text{softmax}\left(\frac{Q_i^T K_j}{\sqrt{d_k}} \cdot P_{\text{effect}}(i,j,L)\right)$$

### Enhanced Position Effect Function

The basic function suffers from information loss at long distances ($e^{-\beta \cdot |i-j|/L} \to 0$). To address this, an enhancement coefficient $\gamma$ is introduced:

$$P_{\text{effect}}(i,j,L) = \alpha \cdot \frac{1 + \gamma \exp\left(-\beta \frac{|i-j|}{L}\right)}{1 + \gamma}$$

Advantages of the enhanced formula:
- **Non-zero lower bound**: Long-distance positions maintain minimum attention weight $\frac{\alpha}{1+\gamma}$
- **Information retention**: Medium-distance information retention improves by 4.2×, maximum distance improves by 28.3×
- **Mathematical properties**: Maintains continuity, differentiability, and monotonicity
- **Minimal overhead**: Memory overhead < 0.1%, training time increase < 1.2%

Default parameters: $\alpha = 1.0$, $\beta = 1.0$, $\gamma = 0.5$

### Mathematical Properties

The position effect function has the following mathematical properties (Theorem 1):
- **Continuity**: Continuous in $|i-j|$, attention changes smoothly with distance
- **Differentiability**: Supports gradient-based optimization and further analysis
- **Monotonicity**: For fixed $\alpha$ and $\beta$, attention decreases monotonically with distance

These properties enable optimal parameter selection (Theorem 2) and convergence results (Theorems 3-5).

---

## Information Importance and Optimal Position

### Information Importance Definition

The information importance of position $j$ is defined as:

$$I_j = \|\mathbf{x}_j\|_2$$

That is, the L2 norm of the position vector, serving as a simple proxy for token importance.

### Position Value Function

The position value function is defined as:

$$V(i) = \sum_j A_{ij} \cdot I_j$$

Representing the total weighted importance aggregated by position $i$ under attention distribution $\{A_{ij}\}_j$. **Information gain** is defined as $V(i)$, i.e., the amount of (importance-weighted) information that position $i$ receives from the sequence.

### Maximum Benefit Position

The **maximum benefit position** is defined as:

$$\text{pos}^* = \arg\max_i V(i)$$

**Proposition 1 (Maximum Benefit Position)**: For finite sequence length $L$, the set $\arg\max_i V(i)$ is non-empty. When $\sum_j A_{ij} = 1$ and $I_j \geq 0$, $V(i)$ is a convex combination of $\{I_j\}$, so any $\text{pos}^* \in \arg\max_i V(i)$ achieves the maximum aggregated importance that a single position can receive under the given attention and importance.

On structured patterns, the consistency between the derived $\text{pos}^*$ and the true position reaches 89%, indicating that this formula connects position with importance in a single, testable way.

---

## Code Structure

```
epar-framework/
├── config.py                          # Unified output directory configuration
├── epar/                              # Core package: position effect and position-aware attention
│   ├── __init__.py
│   └── position_attention_model.py    # Core model implementation
├── experiments/                       # Experiment scripts
│   ├── 1_position_effect/            # Experiment 1: Position effect function
│   │   └── run.py
│   ├── 2_optimal_position/           # Experiment 2: Optimal position and consistency metric
│   │   ├── run.py
│   │   └── optimal_position_verification.py
│   ├── 3_parameter_sensitivity/      # Experiment 3: Parameter sensitivity analysis
│   │   ├── run.py
│   │   ├── parameter_sensitivity_analysis.py
│   │   └── create_improvement_chart.py
│   ├── 4_adaptive_attention/          # Experiment 4: Adaptive attention mechanism
│   │   ├── run.py
│   │   └── adaptive_attention_mechanism.py
│   ├── EXPERIMENTS.md                 # Experiment-paper section mapping table
│   ├── configs/                       # Experiment configurations
│   │   └── paper_protocol.json        # Paper protocol configuration
│   ├── longbench/                    # LongBench benchmark configuration
│   ├── scrolls/                      # SCROLLS benchmark configuration
│   ├── wmt_zh_en/                    # WMT Zh-En configuration
│   ├── codexglue/                    # CodeXGlue configuration
│   └── run_supplementary_experiments.py  # Supplementary experiment runner script
├── output/                           # All experiment outputs (categorized by experiment)
│   ├── position/                     # Position effect analysis results
│   ├── optimal/                      # Optimal position verification results
│   ├── parameter/                    # Parameter sensitivity analysis results
│   └── adaptive/                     # Adaptive attention analysis results
├── scripts/                          # Utility scripts
│   ├── migrate_outputs.py            # Migrate old output directories
│   ├── check_length.py               # Length checking utility
│   └── README.md
├── run_all.py                        # One-click runner for all analysis experiments
├── requirements.txt                  # Dependency package list
└── README.md                         # This document
```

---

## Experiment Description

### Experiment-Paper Correspondence

| Experiment | Paper Section | Content Summary | Output Directory |
|-----------|---------------|-----------------|------------------|
| **1_position_effect** | **Section 4** Position Effect Function<br>Appendix: Mathematical Characteristics, Attention Result Diagram | Position effect function $P_{\text{effect}}(i,j,L)$, mathematical properties (continuity, differentiability, monotonicity), attention statistics and heatmaps | `output/position/` |
| **2_optimal_position** | **Section 4.3** Information Importance & Optimal Position<br>Appendix: Metric for Maximum Benefit Position, Enhanced Results Analysis | Information importance $I_j$, optimal position $\text{pos}^* = \arg\max_i V(i)$, consistency metric and ranking correlation, results for various information distributions under enhanced formula | `output/optimal/` |
| **3_parameter_sensitivity** | **Appendix** Systematic Ablation Study, Gamma Parameter Optimization, Parameter Selection Theory | $\alpha,\beta,\gamma$ sensitivity analysis, Basic vs Enhanced comparison, ablation study and improvement curves | `output/parameter/` |
| **4_adaptive_attention** | **Section 5.4** When and Why the Method Helps | Adaptive/task-aware attention, when to use EPAR, parameter and task type analysis | `output/adaptive/` |

Detailed correspondence can be found in [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md).

### Experiment 1: Position Effect Function

**Paper Section**: Section 4, Appendix (Mathematical Characteristics)

**Functionality**:
- Implement position effect function $P_{\text{effect}}(i,j,L)$
- Analyze mathematical properties (continuity, differentiability, monotonicity)
- Generate attention weight distribution heatmaps
- Visualize position effect matrix

**Run**:
```bash
python experiments/1_position_effect/run.py
```

**Output**:
- `output/position/attention_weights_heatmap.png` - Attention weights heatmap
- `output/position/position_effect_matrix.png` - Position effect matrix
- `output/position/position_correlation.png` - Position correlation matrix
- `output/position/average_attention_distribution.png` - Average attention distribution
- `output/position/information_importance_distribution.png` - Information importance distribution
- `output/position/position_score_distribution.png` - Position score distribution
- `output/position/attention_statistics.png` - Attention statistics
- `output/position/position_effect_analysis.png` - Position effect analysis
- `output/position/position_attention_analysis.png` - Position attention analysis

### Experiment 2: Optimal Position and Consistency Metric

**Paper Section**: Section 4.3, Appendix (Metric for Maximum Benefit Position)

**Functionality**:
- Calculate information importance $I_j = \|\mathbf{x}_j\|_2$
- Derive maximum benefit position $\text{pos}^* = \arg\max_i V(i)$
- Verify consistency metric
- Calculate ranking correlation
- Analyze performance under different information distribution patterns

**Run**:
```bash
python experiments/2_optimal_position/run.py
```

**Output**:
- `output/optimal/verification_results.png` - Verification results visualization
- `output/optimal/consistency_verification.png` - Consistency verification
- `output/optimal/position_score_analysis.png` - Position score analysis
- `output/optimal/comprehensive_performance_score.png` - Comprehensive performance score
- `output/optimal/consistency_vs_ranking_correlation.png` - Consistency vs ranking correlation
- `output/optimal/information_type_radar_comparison.png` - Information type radar comparison
- `output/optimal/standard_deviation_analysis.png` - Standard deviation analysis
- `output/optimal/verification_report.txt` - Verification report

### Experiment 3: Parameter Sensitivity Analysis

**Paper Section**: Appendix (Systematic Ablation Study, Parameter Selection Theory)

**Functionality**:
- Analyze $\alpha$ parameter sensitivity (position influence strength)
- Analyze $\beta$ parameter sensitivity (spatial decay rate)
- Analyze $\gamma$ parameter sensitivity (long-distance enhancement)
- Basic vs Enhanced formula comparison
- Parameter combination heatmaps
- Performance improvement curves

**Run**:
```bash
python experiments/3_parameter_sensitivity/run.py
```

**Output**:
- `output/parameter/alpha_parameter_sensitivity.png` - α parameter sensitivity
- `output/parameter/beta_parameter_sensitivity.png` - β parameter sensitivity
- `output/parameter/combined_parameter_sensitivity.png` - Combined parameter sensitivity
- `output/parameter/parameter_combination_heatmap.png` - Parameter combination heatmap
- `output/parameter/enhanced_performance_analysis.png` - Enhanced performance analysis
- `output/parameter/position_correlation_comparison.png` - Position correlation comparison
- `output/parameter/alpha_parameter_data.csv` - α parameter data
- `output/parameter/beta_parameter_data.csv` - β parameter data
- `output/parameter/parameter_sensitivity_report.txt` - Parameter sensitivity report

### Experiment 4: Adaptive Attention Mechanism

**Paper Section**: Section 5.4 When and Why the Method Helps

**Functionality**:
- Task-aware attention
- Content-aware attention
- Adaptive fusion mechanism
- Analyze when to use EPAR method
- Task type and parameter relationship analysis

**Run**:
```bash
python experiments/4_adaptive_attention/run.py
```

**Output**:
- `output/adaptive/adaptive_attention_analysis.png` - Adaptive attention analysis
- `output/adaptive/attention_weights_heatmap.png` - Attention weights heatmap
- `output/adaptive/attention_entropy_distribution.png` - Attention entropy distribution
- `output/adaptive/task_weights_distribution.png` - Task weights distribution
- `output/adaptive/content_importance_distribution.png` - Content importance distribution
- `output/adaptive/task_type_distribution.png` - Task type distribution
- `output/adaptive/content_type_distribution.png` - Content type distribution
- `output/adaptive/position_attention_analysis.png` - Position attention analysis
- `output/adaptive/parameter_sensitivity_*.png` - Parameter sensitivity analysis plots

---

## Quick Start

### Environment Requirements

- Python >= 3.7
- PyTorch >= 1.8.0
- NumPy
- Matplotlib
- Seaborn

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Experiments

#### Method 1: Run by Paper Experiment Number (Recommended)

```bash
# Experiment 1: Position effect function
python experiments/1_position_effect/run.py

# Experiment 2: Optimal position and consistency metric
python experiments/2_optimal_position/run.py

# Experiment 3: Parameter sensitivity analysis
python experiments/3_parameter_sensitivity/run.py

# Experiment 4: Adaptive attention mechanism
python experiments/4_adaptive_attention/run.py
```

#### Method 2: One-Click Run All Analysis Experiments

```bash
python run_all.py
```

### Supplementary Experiments

Supplementary experiments correspond to Paper Section 5, including LongBench, SCROLLS, WMT Zh-En, CodeXGlue, and other benchmark tests.

```bash
cd experiments

# LongBench (long-context benchmark)
python run_supplementary_experiments.py --benchmark longbench [--subset narrativeqa,qasper] [--quick]

# SCROLLS (long-document tasks)
python run_supplementary_experiments.py --benchmark scrolls [--task gov_report] [--quick]

# WMT Zh-En (non-English translation)
python run_supplementary_experiments.py --benchmark wmt_zh_en

# CodeXGlue (code tasks)
python run_supplementary_experiments.py --benchmark codexglue
```

Supplementary experiment results are saved in the `experiments/results/<benchmark>/` directory.

---

## Output Description

All experiment outputs are uniformly saved in the `output/` directory, categorized by experiment type:

- `output/position/` - Position effect function analysis results
  - Attention weights heatmaps
  - Position effect matrices
  - Position correlation matrices
  - Statistical analysis and visualizations

- `output/optimal/` - Optimal position verification results
  - Consistency verification results
  - Position score analysis
  - Performance comparison under different information distribution patterns
  - Verification reports

- `output/parameter/` - Parameter sensitivity analysis results
  - $\alpha$, $\beta$, $\gamma$ parameter sensitivity curves
  - Parameter combination heatmaps
  - Basic vs Enhanced comparison
  - Performance improvement analysis

- `output/adaptive/` - Adaptive attention analysis results
  - Task-aware attention analysis
  - Content-aware attention analysis
  - Attention entropy distribution
  - Parameter sensitivity analysis

- `experiments/results/` - Supplementary experiment results
  - LongBench, SCROLLS, WMT, CodeXGlue, and other benchmark test results

---

## Core Implementation

### Position Effect Function Class

```python
class PositionEffectFunction:
    """Position Effect Function Class"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 1.5):
        self.alpha = alpha  # Position influence strength parameter
        self.beta = beta    # Position decay parameter
        self.gamma = gamma  # Enhancement coefficient (Enhanced formula)
    
    def __call__(self, i: int, j: int, L: int) -> float:
        """
        Calculate enhanced position effect function
        
        Enhanced formula: P_effect = α * (1 + γ * e^(-β * |i-j|/L)) / (1 + γ)
        """
        distance = abs(i - j)
        normalized_distance = distance / L
        
        base_effect = math.exp(-self.beta * normalized_distance)
        enhanced_effect = (1 + self.gamma * base_effect) / (1 + self.gamma)
        
        return self.alpha * enhanced_effect
```

### Position-Aware Attention Module

```python
class PositionAwareAttention(nn.Module):
    """Position-Aware Attention Mechanism"""
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Calculate standard attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hidden_dim)
        
        # Apply position effect function
        position_weights = self.position_effect.get_position_matrix(seq_len)
        attention_scores = attention_scores * position_weights
        
        # Softmax normalization
        attention_weights = F.softmax(attention_scores / temperature, dim=-1)
        
        return output, attention_weights
```

---

## Experimental Results Summary

### Main Performance Improvements

| Task | Best Baseline | EPAR (Basic) | EPAR (Enhanced) | Improvement |
|------|---------------|--------------|-----------------|-------------|
| WikiText-103 (PPL↓) | 23.5±0.20 | 23.2±0.15 | **22.8±0.12** | **4.7%** |
| WMT'14 En-De (BLEU↑) | 29.1±0.30 | 29.3±0.25 | **29.6±0.20** | **1.7%** |
| SQuAD 2.0 (F1↑) | 0.831±0.004 | 0.835±0.003 | **0.842±0.003** | **2.4%** |
| GLUE (Acc↑) | 0.852±0.004 | 0.856±0.003 | **0.861±0.003** | **1.8%** |
| ArXiv (ROUGE-L↑) | 0.439±0.004 | 0.445±0.003 | **0.462±0.003** | **8.9%** |

### Consistency Metric

- **Structured data**: Consistency 0.9063 → 0.934 (Enhanced), ranking correlation 0.5932 → 0.678
- **Clustered data**: Consistency 0.8543 → 0.891, ranking correlation 0.2390 → 0.387
- **All patterns**: Consistency maintained above 0.7

### Information Retention

- **Medium distance**: Information retention improves by 4.2×
- **Maximum distance**: Information retention improves by 28.3× (Enhanced formula)
- **Maximum distance information retention rate**: 78% (vs. 2.8% for basic formula)

---

## When to Use EPAR

### Recommended Scenarios for EPAR

1. **Long sequences** (> 512 tokens): Long documents, ArXiv papers, etc. (+8.9% ROUGE-L)
2. **Retrieval and long-context QA**: $\gamma$ lower bound preserves long-distance evidence
3. **Structured/clustered importance**: Consistency 0.89-0.93
4. **Translation, QA, dialogue**: Tasks with positional structure (+1.7%-2.4%)

### Not Recommended Scenarios for EPAR

1. **Short sequences** (< 256 tokens): Limited benefit (1.2%-1.8%)
2. **Random or shuffled order**: e.g., shuffled SQuAD F1 decreases by 2.5%
3. **Non-sequential tasks**: Set, graph operations, etc., small benefit with added overhead
4. **High-noise data**: Recommend reducing $\alpha$ or using standard attention

---

## Theoretical Contributions

1. **EPAR Framework**: Explicitly model position-attention relationships at the attention score level
2. **Mathematical Analysis**: Prove continuity, differentiability, monotonicity (Theorem 1)
3. **Optimal Parameter Selection**: Theoretical guarantees (Theorem 2)
4. **Convergence Proofs**: Convergence properties (Theorems 3-5)
5. **Enhanced Formula**: Introduce $\gamma$ coefficient to address long-distance information loss

---

## Theoretical Comparison with Existing Methods

### Operation Level

- **Existing methods**: Operate at vector representation level (RoPE, Shaw, Transformer-XL) or add fixed bias (ALiBi)
- **EPAR**: Operate at attention score level, modeling through explicit functions

### Position-Attention Relationship

- **Existing methods**: Implicit relationship, difficult to analyze
- **EPAR**: Explicit relationship, derivable optimal positions

### Controllability

- **Existing methods**: Fixed or learnable, but lack direct control
- **EPAR**: Fine-grained control through $\alpha$, $\beta$, $\gamma$

### Theoretical Guarantees

- **Existing methods**: Lack theoretical guarantees
- **EPAR**: Optimal parameter selection (Theorem 2) and convergence properties (Theorems 3-5)

---

## Experimental Protocol

All experiments follow the paper protocol (`experiments/configs/paper_protocol.json`):

- **Model scale**: ~110M parameters (12 layers, 768 hidden dimensions, 12 heads)
- **Random seeds**: 42-46 (5 runs)
- **Baseline methods**: RoPE, ALiBi, Shaw, Transformer-XL
- **Evaluation**: Bonferroni correction ($p < 0.01$), effect size (Cohen's d)
- **Default hyperparameters**: $\alpha = 1.0$, $\beta = 1.0$, $\gamma = 0.5$

---

## Citation

If you use this codebase, please cite our paper:

```bibtex
@article{epar2026,
  title={EPAR: Explicit Position-Attention Relationship for Interpretable Long-Range Attention in Transformers},
  author={Wang, Weiwei},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project follows the corresponding open-source license. See LICENSE file for details.

---

## Contact

For questions or suggestions, please contact:
- Author: Weiwei Wang
- Email: weiweiw404@gmail.com
- Institution: Sunline Technology Co., Ltd.

---

## Acknowledgments

Thanks to all researchers and developers who have contributed to this project.
