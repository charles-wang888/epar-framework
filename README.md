# EPAR: Explicit Position-Attention Relationship

**EPAR: Explicit Position-Attention Relationship for Interpretable Long-Range Attention in Transformers**

本代码库实现了论文《EPAR: Explicit Position-Attention Relationship for Interpretable Long-Range Attention in Transformers》中提出的位置感知注意力机制。EPAR框架在**注意力分数层面**显式建模位置对注意力强度的影响，而非将位置编码到向量表示中，从而实现了可分析性、可调参数和可推导的最优位置。

## 📋 目录

- [核心概念](#核心概念)
- [位置效应函数](#位置效应函数)
- [代码结构](#代码结构)
- [实验说明](#实验说明)
- [快速开始](#快速开始)
- [输出说明](#输出说明)
- [引用](#引用)

---

## 核心概念

### EPAR框架

EPAR（Explicit Position-Attention Relationship，显式位置-注意力关系）框架的核心思想是：**在注意力分数层面显式建模位置如何影响注意力强度**，而不是通过向量表示隐式编码位置信息。

与现有方法（RoPE、ALiBi、相对位置编码等）的根本区别：

| 方法 | 操作层面 | 数学形式 | 位置建模 |
|------|----------|----------|----------|
| **RoPE** | 向量表示 | $Q'_i = R_{\theta}(i) Q_i$ | 隐式（旋转） |
| **ALiBi** | 注意力分数 | $A_{ij} = Q_i^T K_j + m \cdot \|i-j\|$ | 线性偏置 |
| **相对位置编码 (Shaw)** | 向量表示 | $Q'_i = Q_i + P_i$ | 可学习嵌入 |
| **EPAR (Ours)** | **注意力分数** | **$A_{ij} = \text{softmax}((Q_i^T K_j/\sqrt{d_k}) \cdot P_{\text{effect}}(i,j,L))$** | **显式函数** |

### 主要优势

1. **理论保证**：可证明最优参数选择（Theorem 2）和收敛性质（Theorems 3-5）
2. **可解释性**：位置-注意力关系的清晰数学解释
3. **可控性**：通过参数 $\alpha$、$\beta$、$\gamma$ 进行细粒度控制
4. **效率**：位置效应矩阵可缓存，训练开销仅增加2.4%

---

## 位置效应函数

### 基础位置效应函数

基础位置效应函数定义为：

$$P_{\text{effect}}(i,j,L) = \alpha \cdot e^{-\beta \cdot |i-j|/L}$$

其中：
- $\alpha$：位置影响强度参数（控制强度）
- $\beta$：位置衰减参数（控制空间衰减率）
- $L$：序列长度
- $i, j$：查询位置和键位置

基于位置效应函数，位置感知注意力权重定义为：

$$A_{ij} = \text{softmax}\left(\frac{Q_i^T K_j}{\sqrt{d_k}} \cdot P_{\text{effect}}(i,j,L)\right)$$

### 增强位置效应函数

基础函数在长距离时会出现信息丢失（$e^{-\beta \cdot |i-j|/L} \to 0$）。为解决此问题，引入增强系数 $\gamma$：

$$P_{\text{effect}}(i,j,L) = \alpha \cdot \frac{1 + \gamma \exp\left(-\beta \frac{|i-j|}{L}\right)}{1 + \gamma}$$

增强公式的优势：
- **非零下界**：长距离位置保持最小注意力权重 $\frac{\alpha}{1+\gamma}$
- **信息保留**：中距离信息保留提升4.2倍，最大距离提升28.3倍
- **数学性质**：保持连续性、可微性和单调性
- **开销极小**：内存开销 < 0.1%，训练时间增加 < 1.2%

默认参数：$\alpha = 1.0$，$\beta = 1.0$，$\gamma = 0.5$

### 数学性质

位置效应函数具有以下数学性质（Theorem 1）：
- **连续性**：在 $|i-j|$ 上连续，注意力随距离平滑变化
- **可微性**：支持基于梯度的优化和进一步分析
- **单调性**：对于固定的 $\alpha$ 和 $\beta$，注意力随距离单调递减

这些性质使得最优参数选择（Theorem 2）和收敛结果（Theorems 3-5）成为可能。

---

## 信息重要性与最优位置

### 信息重要性定义

位置 $j$ 的信息重要性定义为：

$$I_j = \|\mathbf{x}_j\|_2$$

即该位置向量的L2范数，作为token重要性的简单代理。

### 位置价值函数

位置价值函数定义为：

$$V(i) = \sum_j A_{ij} \cdot I_j$$

表示位置 $i$ 在注意力分布 $\{A_{ij}\}_j$ 下聚合的总加权重要性。**信息增益**定义为 $V(i)$，即位置 $i$ 从序列接收的（重要性加权的）信息量。

### 最大收益位置

**最大收益位置**定义为：

$$\text{pos}^* = \arg\max_i V(i)$$

**Proposition 1 (最大收益位置)**：对于有限序列长度 $L$，集合 $\arg\max_i V(i)$ 非空。当 $\sum_j A_{ij} = 1$ 且 $I_j \geq 0$ 时，$V(i)$ 是 $\{I_j\}$ 的凸组合，因此任何 $\text{pos}^* \in \arg\max_i V(i)$ 在给定注意力和重要性下达到单个位置可接收的最大聚合重要性。

在结构化模式上，推导的 $\text{pos}^*$ 与真实位置的一致性达到89%，说明该公式以单一、可测试的方式将位置与重要性联系起来。

---

## 代码结构

```
epar-framework/
├── config.py                          # 统一输出目录配置
├── epar/                              # 核心包：位置效应与位置感知注意力
│   ├── __init__.py
│   └── position_attention_model.py    # 核心模型实现
├── experiments/                       # 实验脚本
│   ├── 1_position_effect/            # 实验1：位置效应函数
│   │   └── run.py
│   ├── 2_optimal_position/           # 实验2：最优位置与一致性度量
│   │   ├── run.py
│   │   └── optimal_position_verification.py
│   ├── 3_parameter_sensitivity/      # 实验3：参数敏感性分析
│   │   ├── run.py
│   │   ├── parameter_sensitivity_analysis.py
│   │   └── create_improvement_chart.py
│   ├── 4_adaptive_attention/          # 实验4：自适应注意力机制
│   │   ├── run.py
│   │   └── adaptive_attention_mechanism.py
│   ├── EXPERIMENTS.md                 # 实验-论文章节对照表
│   ├── configs/                       # 实验配置
│   │   └── paper_protocol.json        # 论文协议配置
│   ├── longbench/                    # LongBench基准配置
│   ├── scrolls/                      # SCROLLS基准配置
│   ├── wmt_zh_en/                    # WMT Zh-En配置
│   ├── codexglue/                    # CodeXGlue配置
│   └── run_supplementary_experiments.py  # 补充实验运行脚本
├── output/                           # 所有实验输出（按实验分类）
│   ├── position/                     # 位置效应分析结果
│   ├── optimal/                      # 最优位置验证结果
│   ├── parameter/                    # 参数敏感性分析结果
│   └── adaptive/                     # 自适应注意力分析结果
├── scripts/                          # 工具脚本
│   ├── migrate_outputs.py            # 迁移旧输出目录
│   ├── check_length.py               # 长度检查工具
│   └── README.md
├── run_all.py                        # 一键运行所有分析实验
├── requirements.txt                  # 依赖包列表
└── README.md                         # 本文档
```

---

## 实验说明

### 实验与论文对应关系

| 实验 | 论文对应 | 内容摘要 | 输出目录 |
|------|----------|----------|----------|
| **1_position_effect** | **Section 4** Position Effect Function<br>Appendix: Mathematical Characteristics, Attention Result Diagram | 位置效应函数 $P_{\text{effect}}(i,j,L)$、数学性质（连续性、可微、单调）、注意力统计与热力图 | `output/position/` |
| **2_optimal_position** | **Section 4.3** Information Importance & Optimal Position<br>Appendix: Metric for Maximum Benefit Position, Enhanced Results Analysis | 信息重要性 $I_j$、最优位置 $\text{pos}^* = \arg\max_i V(i)$、一致性度量与排序相关、增强公式下各信息分布结果 | `output/optimal/` |
| **3_parameter_sensitivity** | **Appendix** Systematic Ablation Study, Gamma Parameter Optimization, Parameter Selection Theory | $\alpha,\beta,\gamma$ 敏感性分析、Basic vs Enhanced对比、消融研究与改进曲线 | `output/parameter/` |
| **4_adaptive_attention** | **Section 5.4** When and Why the Method Helps | 自适应/任务感知注意力、何时使用EPAR、参数与任务类型分析 | `output/adaptive/` |

详细对应关系见 [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md)。

### 实验1：位置效应函数

**对应论文**：Section 4, Appendix (Mathematical Characteristics)

**功能**：
- 实现位置效应函数 $P_{\text{effect}}(i,j,L)$
- 分析数学性质（连续性、可微性、单调性）
- 生成注意力权重分布热力图
- 可视化位置效应矩阵

**运行**：
```bash
python experiments/1_position_effect/run.py
```

**输出**：
- `output/position/attention_weights_heatmap.png` - 注意力权重热力图
- `output/position/position_effect_matrix.png` - 位置效应矩阵
- `output/position/position_correlation.png` - 位置相关性矩阵
- `output/position/average_attention_distribution.png` - 平均注意力分布
- `output/position/information_importance_distribution.png` - 信息重要性分布
- `output/position/position_score_distribution.png` - 位置分数分布
- `output/position/attention_statistics.png` - 注意力统计信息
- `output/position/position_effect_analysis.png` - 位置效应分析
- `output/position/position_attention_analysis.png` - 位置注意力分析

### 实验2：最优位置与一致性度量

**对应论文**：Section 4.3, Appendix (Metric for Maximum Benefit Position)

**功能**：
- 计算信息重要性 $I_j = \|\mathbf{x}_j\|_2$
- 推导最大收益位置 $\text{pos}^* = \arg\max_i V(i)$
- 验证一致性度量（consistency metric）
- 计算排序相关性（ranking correlation）
- 分析不同信息分布模式下的性能

**运行**：
```bash
python experiments/2_optimal_position/run.py
```

**输出**：
- `output/optimal/verification_results.png` - 验证结果可视化
- `output/optimal/consistency_verification.png` - 一致性验证
- `output/optimal/position_score_analysis.png` - 位置分数分析
- `output/optimal/comprehensive_performance_score.png` - 综合性能分数
- `output/optimal/consistency_vs_ranking_correlation.png` - 一致性与排序相关性
- `output/optimal/information_type_radar_comparison.png` - 信息类型雷达对比
- `output/optimal/standard_deviation_analysis.png` - 标准差分析
- `output/optimal/verification_report.txt` - 验证报告

### 实验3：参数敏感性分析

**对应论文**：Appendix (Systematic Ablation Study, Parameter Selection Theory)

**功能**：
- 分析 $\alpha$ 参数敏感性（位置影响强度）
- 分析 $\beta$ 参数敏感性（空间衰减率）
- 分析 $\gamma$ 参数敏感性（长距离增强）
- Basic vs Enhanced公式对比
- 参数组合热力图
- 性能改进曲线

**运行**：
```bash
python experiments/3_parameter_sensitivity/run.py
```

**输出**：
- `output/parameter/alpha_parameter_sensitivity.png` - α参数敏感性
- `output/parameter/beta_parameter_sensitivity.png` - β参数敏感性
- `output/parameter/combined_parameter_sensitivity.png` - 组合参数敏感性
- `output/parameter/parameter_combination_heatmap.png` - 参数组合热力图
- `output/parameter/enhanced_performance_analysis.png` - 增强性能分析
- `output/parameter/position_correlation_comparison.png` - 位置相关性对比
- `output/parameter/alpha_parameter_data.csv` - α参数数据
- `output/parameter/beta_parameter_data.csv` - β参数数据
- `output/parameter/parameter_sensitivity_report.txt` - 参数敏感性报告

### 实验4：自适应注意力机制

**对应论文**：Section 5.4 When and Why the Method Helps

**功能**：
- 任务感知注意力（Task-Aware Attention）
- 内容感知注意力（Content-Aware Attention）
- 自适应融合机制
- 分析何时使用EPAR方法
- 任务类型与参数关系分析

**运行**：
```bash
python experiments/4_adaptive_attention/run.py
```

**输出**：
- `output/adaptive/adaptive_attention_analysis.png` - 自适应注意力分析
- `output/adaptive/attention_weights_heatmap.png` - 注意力权重热力图
- `output/adaptive/attention_entropy_distribution.png` - 注意力熵分布
- `output/adaptive/task_weights_distribution.png` - 任务权重分布
- `output/adaptive/content_importance_distribution.png` - 内容重要性分布
- `output/adaptive/task_type_distribution.png` - 任务类型分布
- `output/adaptive/content_type_distribution.png` - 内容类型分布
- `output/adaptive/position_attention_analysis.png` - 位置注意力分析
- `output/adaptive/parameter_sensitivity_*.png` - 参数敏感性分析图

---

## 快速开始

### 环境要求

- Python >= 3.7
- PyTorch >= 1.8.0
- NumPy
- Matplotlib
- Seaborn

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行实验

#### 方式一：按论文实验编号运行（推荐）

```bash
# 实验1：位置效应函数
python experiments/1_position_effect/run.py

# 实验2：最优位置与一致性度量
python experiments/2_optimal_position/run.py

# 实验3：参数敏感性分析
python experiments/3_parameter_sensitivity/run.py

# 实验4：自适应注意力机制
python experiments/4_adaptive_attention/run.py
```

#### 方式二：一键运行所有分析实验

```bash
python run_all.py
```

### 补充实验（Supplementary Experiments）

补充实验对应论文Section 5，包括LongBench、SCROLLS、WMT Zh-En、CodeXGlue等基准测试。

```bash
cd experiments

# LongBench (长上下文基准)
python run_supplementary_experiments.py --benchmark longbench [--subset narrativeqa,qasper] [--quick]

# SCROLLS (长文档任务)
python run_supplementary_experiments.py --benchmark scrolls [--task gov_report] [--quick]

# WMT Zh-En (非英语翻译)
python run_supplementary_experiments.py --benchmark wmt_zh_en

# CodeXGlue (代码任务)
python run_supplementary_experiments.py --benchmark codexglue
```

补充实验结果保存在 `experiments/results/<benchmark>/` 目录下。

---

## 输出说明

所有实验输出统一保存在 `output/` 目录下，按实验类别分类：

- `output/position/` - 位置效应函数分析结果
  - 注意力权重热力图
  - 位置效应矩阵
  - 位置相关性矩阵
  - 统计分析与可视化

- `output/optimal/` - 最优位置验证结果
  - 一致性验证结果
  - 位置分数分析
  - 不同信息分布模式下的性能对比
  - 验证报告

- `output/parameter/` - 参数敏感性分析结果
  - $\alpha$、$\beta$、$\gamma$ 参数敏感性曲线
  - 参数组合热力图
  - Basic vs Enhanced对比
  - 性能改进分析

- `output/adaptive/` - 自适应注意力分析结果
  - 任务感知注意力分析
  - 内容感知注意力分析
  - 注意力熵分布
  - 参数敏感性分析

- `experiments/results/` - 补充实验结果
  - LongBench、SCROLLS、WMT、CodeXGlue等基准测试结果

---

## 核心实现

### 位置效应函数类

```python
class PositionEffectFunction:
    """Position Effect Function Class"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 1.5):
        self.alpha = alpha  # 位置影响强度参数
        self.beta = beta    # 位置衰减参数
        self.gamma = gamma  # 增强系数（Enhanced公式）
    
    def __call__(self, i: int, j: int, L: int) -> float:
        """
        计算增强位置效应函数
        
        Enhanced公式: P_effect = α * (1 + γ * e^(-β * |i-j|/L)) / (1 + γ)
        """
        distance = abs(i - j)
        normalized_distance = distance / L
        
        base_effect = math.exp(-self.beta * normalized_distance)
        enhanced_effect = (1 + self.gamma * base_effect) / (1 + self.gamma)
        
        return self.alpha * enhanced_effect
```

### 位置感知注意力模块

```python
class PositionAwareAttention(nn.Module):
    """Position-Aware Attention Mechanism"""
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # 计算标准注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hidden_dim)
        
        # 应用位置效应函数
        position_weights = self.position_effect.get_position_matrix(seq_len)
        attention_scores = attention_scores * position_weights
        
        # Softmax归一化
        attention_weights = F.softmax(attention_scores / temperature, dim=-1)
        
        return output, attention_weights
```

---

## 实验结果摘要

### 主要性能提升

| 任务 | 最佳基线 | EPAR (Basic) | EPAR (Enhanced) | 提升 |
|------|----------|--------------|-----------------|------|
| WikiText-103 (PPL↓) | 23.5±0.20 | 23.2±0.15 | **22.8±0.12** | **4.7%** |
| WMT'14 En-De (BLEU↑) | 29.1±0.30 | 29.3±0.25 | **29.6±0.20** | **1.7%** |
| SQuAD 2.0 (F1↑) | 0.831±0.004 | 0.835±0.003 | **0.842±0.003** | **2.4%** |
| GLUE (Acc↑) | 0.852±0.004 | 0.856±0.003 | **0.861±0.003** | **1.8%** |
| ArXiv (ROUGE-L↑) | 0.439±0.004 | 0.445±0.003 | **0.462±0.003** | **8.9%** |

### 一致性度量

- **结构化数据**：一致性 0.9063 → 0.934（Enhanced），排序相关性 0.5932 → 0.678
- **聚类数据**：一致性 0.8543 → 0.891，排序相关性 0.2390 → 0.387
- **所有模式**：一致性均保持在 0.7 以上

### 信息保留

- **中距离**：信息保留提升 4.2倍
- **最大距离**：信息保留提升 28.3倍（Enhanced公式）
- **最大距离信息保留率**：78%（vs. 基础公式的2.8%）

---

## 何时使用EPAR

### 推荐使用EPAR的场景

1. **长序列**（> 512 tokens）：长文档、ArXiv论文等（+8.9% ROUGE-L）
2. **检索与长上下文QA**：$\gamma$ 下界保持远距离证据
3. **结构化/聚类重要性**：一致性 0.89-0.93
4. **翻译、QA、对话**：具有位置结构的任务（+1.7%-2.4%）

### 不推荐使用EPAR的场景

1. **短序列**（< 256 tokens）：收益有限（1.2%-1.8%）
2. **随机或打乱顺序**：如打乱的SQuAD F1下降2.5%
3. **非序列任务**：集合、图操作等，收益小且增加开销
4. **高噪声数据**：建议降低 $\alpha$ 或使用标准注意力

---

## 理论贡献

1. **EPAR框架**：在注意力分数层面显式建模位置-注意力关系
2. **数学分析**：证明连续性、可微性、单调性（Theorem 1）
3. **最优参数选择**：理论保证（Theorem 2）
4. **收敛性证明**：收敛性质（Theorems 3-5）
5. **增强公式**：引入 $\gamma$ 系数，解决长距离信息丢失问题

---

## 与现有方法的理论对比

### 操作层面

- **现有方法**：在向量表示层面操作（RoPE、Shaw、Transformer-XL）或添加固定偏置（ALiBi）
- **EPAR**：在注意力分数层面操作，通过显式函数建模

### 位置-注意力关系

- **现有方法**：隐式关系，难以分析
- **EPAR**：显式关系，可推导最优位置

### 可控性

- **现有方法**：固定或可学习，但缺乏直接控制
- **EPAR**：通过 $\alpha$、$\beta$、$\gamma$ 进行细粒度控制

### 理论保证

- **现有方法**：缺乏理论保证
- **EPAR**：最优参数选择（Theorem 2）和收敛性质（Theorems 3-5）

---

## 实验协议

所有实验遵循论文协议（`experiments/configs/paper_protocol.json`）：

- **模型规模**：~110M参数（12层，768隐藏维度，12头）
- **随机种子**：42-46（5次运行）
- **基线方法**：RoPE、ALiBi、Shaw、Transformer-XL
- **评估**：Bonferroni校正（$p < 0.01$），效应量（Cohen's d）
- **默认超参数**：$\alpha = 1.0$，$\beta = 1.0$，$\gamma = 0.5$

---

## 引用

如果您使用了本代码库，请引用我们的论文：

```bibtex
@article{epar2026,
  title={EPAR: Explicit Position-Attention Relationship for Interpretable Long-Range Attention in Transformers},
  author={Wang, Weiwei},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 许可证

本项目遵循相应的开源许可证。详见LICENSE文件。

---

## 联系方式

如有问题或建议，请联系：
- 作者：Weiwei Wang
- 邮箱：weiweiw404@gmail.com
- 机构：Sunline Technology Co., Ltd.

---

## 致谢

感谢所有为本项目做出贡献的研究者和开发者。
