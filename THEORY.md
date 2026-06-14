# D2 Spectral-Aware Quantization Planner — Mathematical Theory

**Complete theoretical foundation combining Random Matrix Theory, Graph Optimization, and CUDA stability modeling.**

---

## Table of Contents

1. [Overview](#overview)
2. [Family (A): Spectral / Random Matrix Theory](#family-a-spectral--random-matrix-theory)
3. [Family (B): Graph Optimization](#family-b-graph-optimization)
4. [Family (C): CUDA Stability Cost Model](#family-c-cuda-stability-cost-model)
5. [Family (D): Quantization Policy Engine](#family-d-quantization-policy-engine)
6. [C-Vector: Information-Theoretic Descriptor](#c-vector-information-theoretic-descriptor)
7. [Integration: Spectral IR Compiler](#integration-spectral-ir-compiler)
8. [References](#references)

---

## Overview

D2 combines **four theoretical frameworks**:

```
W (weight matrix)
    ↓
[A] Spectral Analysis (SVD, α_w, entropy)
    ↓
[C-Vector] Information descriptor (C_s, C_i, C_r, C_d)
    ↓
[D] Quantization Policy (instability → dtype)
    ↓
[B] Graph Optimization (edge costs, fragmentation)
    ↓
[C] CUDA Stability (branching ratio, ρ)
    ↓
Q (quantization plan)
```

---

## Family (A): Spectral / Random Matrix Theory

### 1. SVD Decomposition

Every weight matrix can be decomposed:

$$W = U \Sigma V^\top$$

where:
- $U \in \mathbb{R}^{m \times r}$: Left singular vectors
- $\Sigma = \text{diag}(s_1, s_2, \ldots, s_n)$: Singular values, ordered $s_1 \geq s_2 \geq \cdots \geq s_n \geq 0$
- $V^\top \in \mathbb{R}^{r \times n}$: Right singular vectors (transposed)

**Interpretation**: Singular values measure the "energy" along each principal direction.

### 2. Spectral Exponent (α_w)

**Definition**: Assuming power-law decay of singular values:

$$s_i \propto i^{-\beta}$$

Taking logarithms:
$$\log s_i = -\beta \log i + c$$

**Estimation** (linear regression):

$$\beta = -\frac{d \log s_i}{d \log i} = -\text{polyfit}(\log i, \log s_i)_1$$

**D2 convention**:
$$\alpha_w = 2\beta$$

**Interpretation**:
- **High α_w** (≈ 2.5): Rapid decay → highly compressible
- **Low α_w** (≈ 0.5): Slow decay → full-rank structure

**References**:
- Martin & Mahoney (2017): "Implicit Self-Regularization in Deep Neural Networks"
- Pennington et al. (2017): "Dynamical Isometry and a Mean Field Theory of CNNs"

### 3. Stable Rank

**Definition**:
$$r_s(W) = \frac{\|W\|_F^2}{\|W\|_2^2} = \frac{\sum_i s_i^2}{s_1^2}$$

where:
- $\|W\|_F = \sqrt{\sum_i s_i^2}$ (Frobenius norm)
- $\|W\|_2 = s_1$ (spectral norm)

**Interpretation**:
- Measures effective dimensionality
- $r_s = 1$: pure rank-1 (maximally compressible)
- $r_s = \min(m,n)$: full rank (hard to compress)

### 4. Participation Ratio (Effective Rank)

**Definition**:
$$r_{eff} = \frac{\left(\sum_i s_i\right)^2}{\sum_i s_i^2}$$

**Interpretation**: How many singular values are "active"?

- $r_{eff} \approx 1$: One dominant component
- $r_{eff} \approx n$: Uniform distribution (full-rank)

### 5. Spectral Entropy

**Normalization** (probability distribution):
$$p_i = \frac{s_i}{\sum_j s_j}$$

**Entropy**:
$$H = -\sum_i p_i \log p_i$$

**Effective rank (entropic)**:
$$r_H = \exp(H)$$

**Interpretation**:
- $H \approx 0$: Concentrated spectrum (low entropy, compressible)
- $H \approx \log(n)$: Uniform spectrum (high entropy, hard to compress)

### 6. Compressibility Proxy

**Definition**:
$$C(W) = \frac{1}{\log(1 + \alpha_w)}$$

**Interpretation**:
- High $\alpha_w$ → $C \to 0.5$ (very compressible)
- Low $\alpha_w$ → $C \to \infty$ (incompressible)

---

## C-Vector: Information-Theoretic Descriptor

### Definition

The **C-vector** is a 4-tuple capturing layer informativeness:

$$C(W) = (C_s, C_i, C_r, C_d)$$

### Components

#### $C_s$ (Spectral Stability)

$$C_s = \frac{1}{\log(1 + \alpha_w)}$$

Measures how stable the spectral structure is under perturbation.

#### $C_i$ (Information Entropy Proxy)

$$C_i = 1 - \frac{H}{\log(r)}$$

where $r$ is the matrix rank.

Measures information concentration (higher = more focused).

#### $C_r$ (Risk / Heavy Tail)

$$C_r = \alpha_w (1 + \rho_d)$$

where $\rho_d = \frac{\text{nnz}(W)}{mn}$ is density.

Measures vulnerability to quantization error (heavy-tail risk).

#### $C_d$ (Density)

$$C_d = \rho_d = \frac{\text{nnz}(W)}{mn}$$

Proportion of non-zero entries.

---

## Family (D): Quantization Policy Engine

### Instability Score

**Definition**:
$$\text{instability} = \alpha_w (1 + \rho_d)$$

**Interpretation**:
- **< 1.2**: Highly stable → can use NVFP4 (aggressive)
- **1.2–1.6**: Moderate → INT8
- **1.6–2.0**: Sensitive → FP8
- **≥ 2.0**: Very sensitive → FP16 (conservative)

### Corrected Quantization Mapping

⚠️ **IMPORTANT**: Physical GPU reality inverts naive assumptions.

**Stable layers** can be **aggressively quantized**.
**Unstable layers** need **more bits**.

$$Q(\text{instability}) = \begin{cases}
\text{NVFP4} & \text{instability} < 1.2 \\
\text{INT8} & 1.2 \leq \text{instability} < 1.6 \\
\text{FP8} & 1.6 \leq \text{instability} < 2.0 \\
\text{FP16} & \text{instability} \geq 2.0
\end{cases}$$

### Ordinal Mapping

For graph algorithms:

$$Q_{\text{ordinal}} = \begin{cases}
0 & \text{NVFP4} \\
1 & \text{INT8} \\
2 & \text{FP8} \\
3 & \text{FP16}
\end{cases}$$

---

## Family (B): Graph Optimization

### Problem Statement

Given $n$ layers with recommended dtypes $Q_1, \ldots, Q_n$:

$$\min \sum_{i=1}^{n-1} C_{\text{edge}}(Q_i, Q_{i+1}) + \lambda \cdot \sum_i C_{\text{stability}}(Q_i)$$

### Edge Cost Function

**Cost of transition** between consecutive layers:

$$C_{\text{edge}}(Q_i, Q_j) = |Q_{\text{ordinal}}(Q_i) - Q_{\text{ordinal}}(Q_j)|$$

**Examples**:
- INT8 → INT8: cost = 0
- INT8 → FP8: cost = 1
- INT8 → FP16: cost = 2
- NVFP4 → FP16: cost = 3

### Fragmentation Metric

**Total fragmentation**:

$$F = \sum_{i=1}^{n-1} C_{\text{edge}}(Q_i, Q_{i+1})$$

Lower fragmentation = fewer dtype changes = better CUDA kernel utilization.

### Break Count

**Major transitions** (cost ≥ 2):

$$B = \sum_{i=1}^{n-1} \mathbb{1}[C_{\text{edge}}(Q_i, Q_{i+1}) \geq 2]$$

Each break requires dtype conversion overhead.

---

## Family (C): CUDA Stability Cost Model

### Spectral Radius

**Definition**:
$$\rho(W) = s_1 = \|W\|_2$$

The largest singular value.

**Critical points**:
- $\rho \ll 1$: Contracting dynamics (information decay)
- $\rho \approx 1.04$: **Edge of chaos** (optimal for computation)
- $\rho > 1.5$: Explosive dynamics

### LIF / Avalanche Model

Neural dynamics in layers:

$$V(t+1) = \lambda V(t) + W S(t) + \xi$$

where:
- $V(t)$: Neuron voltages at time $t$
- $S(t)$: Spike train input
- $\xi$: Noise
- $\lambda$: Leak rate

**Branching ratio** (avalanche statistics):

$$m = \frac{\sum A_{t+1}}{\sum A_t}$$

where $A_t$ = activity (spike count).

**Critical regime**: $m \approx 1$ (scale-free avalanches).

**Connection to $\rho$**: $m \approx \rho$ (eigenvalue-branching equivalence).

### Stability Cost

**Cost function**:
$$\text{Cost} = |\alpha_w - \alpha_*| + \lambda \cdot \text{switch}(Q_i, Q_j)$$

where:
- $\alpha_* \approx 2.2$ (empirically optimal exponent)
- $\text{switch}(Q_i, Q_j)$ = transition cost

**References**:
- Beggs & Plenz (2003): "Neuronal avalanches in neocortical circuits"
- Langton (1990): "Computation at the edge of chaos"

---

## Integration: Spectral IR Compiler

### Intermediate Representation (IR)

For each layer $i$, compute:

$$\text{IR}_i = \{\rho_i, \alpha_{w,i}, H_i, Q_i, \text{instability}_i, \text{cost}_i\}$$

### Global Stability Metric

$$\text{Stability} = e^{-|\rho - 1.04|} \cdot e^{-|\alpha_w - 2.2|}$$

**Interpretation**: Product of Gaussians centered at optimal values.

Maximum at $\rho = 1.04$ and $\alpha_w = 2.2$.

### Optimization Objective

$$\min \sum_{i=1}^n \text{cost}_i + \lambda_1 F + \lambda_2 B - \lambda_3 \text{Stability}$$

where:
- $F$: Fragmentation
- $B$: Break count
- $\text{Stability}$: Global spectral health

### Solver Strategy

1. **Layer-wise analysis**: Compute C-vector for each layer
2. **Policy recommendation**: Map instability → candidate dtype
3. **Graph relaxation**: Smooth over-aggressive or over-conservative dtypes
4. **VRAM budgeting**: Respect total memory constraint
5. **CUDA locality**: Minimize kernel re-specialization

---

## Power-Law Hypothesis Testing

### Kolmogorov-Smirnov Test

**Null hypothesis**: Singular values follow $s_i \sim i^{-\beta}$

**Test statistic**:
$$D = \sup_x |F_{\text{empirical}}(x) - F_{\text{synthetic}}(x)|$$

where $F$ = cumulative distribution.

**Decision**:
- If $p > 0.05$: Accept power-law hypothesis
- If $p < 0.05$: Reject (non-power-law)

**Implications**:
- Power-law → Highly compressible (good for quantization)
- Non-power-law → Possible structural issues

---

## Literature & References

### Spectral Analysis & RMT

1. **Pennington, J., Schoenholz, S., & Ganguli, S.** (2017)
   - *"Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Networks"*
   - Key: Spectral radius ≈ 1 is optimal for signal propagation

2. **Martin, C. H., & Mahoney, M. W.** (2017)
   - *"Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory"*
   - Key: Heavy-tailed self-regularization through power-law spectra

3. **Voiculescu, D. V.** (2005)
   - *"Free Probability and Random Matrices"*
   - Advanced: Free probability limits for spectrum asymptotics

### Graph Optimization

4. **Ahuja, R. K., Magnanti, T. L., & Orlin, J. B.** (1993)
   - *"Network Flows: Theory, Algorithms, and Applications"*
   - Min-cost flow algorithms for edge optimization

### Avalanche Dynamics

5. **Beggs, J. M., & Plenz, D.** (2003)
   - *"Neuronal Avalanches in Neocortical Circuits"*
   - Critical dynamics in biological neural networks

6. **Langton, C. G.** (1990)
   - *"Computation at the Edge of Chaos"*
   - Edge of chaos theory for dynamical systems

### Quantization Theory

7. **Gong, R., et al.** (2019)
   - *"Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks"*
   - Layer-wise sensitivity to quantization

---

## Practical Examples

### Example 1: Stable Layer (High Spectral Exponent)

```
Layer: model.layers.0.mlp.down_proj.weight
Shape: [4096, 14336]
SVD: s = [12.3, 8.1, 5.2, ...]

α_w ≈ 2.8 (steep decay)
Instability = 2.8 × (1 + 0.15) = 3.22
→ Recommendation: FP16 (too sensitive)

C-vector: (0.38, 0.72, 3.22, 0.15)
```

### Example 2: Unstable Layer (Low Spectral Exponent)

```
Layer: model.layers.0.mlp.gate_proj.weight
Shape: [14336, 4096]
SVD: s = [18.2, 17.9, 17.5, ...] (flat)

α_w ≈ 0.6 (slow decay)
Instability = 0.6 × (1 + 0.85) = 1.11
→ Recommendation: NVFP4 (stable, compressible)

C-vector: (0.91, 0.45, 1.11, 0.85)
```

### Example 3: Attention Head (Critical Layer)

```
Layer: model.layers.0.self_attn.q_proj.weight
Shape: [4096, 4096]
SVD: s = [4.1, 3.9, 3.7, ...] (uniform-ish)

α_w ≈ 1.2
Instability = 1.2 × 1.1 = 1.32
→ Recommendation: INT8

ρ ≈ 4.1 (slightly supercritical)
→ Monitor for instability
```

---

## Implementation Checklist

- [ ] Compute SVD for all weights
- [ ] Estimate α_w via log-log fit
- [ ] Build C-vectors
- [ ] Recommend dtypes via instability threshold
- [ ] Run KS power-law test
- [ ] Optimize graph fragmentation
- [ ] Validate VRAM constraint
- [ ] Export JSON quantization plan
- [ ] Generate llama.cpp command

---

## FAQ

**Q: Why α_w = 2β and not just β?**
A: Convention in spectral literature. Sometimes α is defined as the exponent directly; here we use 2β to match matrix norm scalings.

**Q: What's the optimal ρ?**
A: ρ ≈ 1.04 (edge of chaos). Too low = decaying signals; too high = explosive.

**Q: Can I override quantization recommendations?**
A: Yes! The policy provides suggestions; graph optimization allows relaxation under VRAM constraints.

**Q: Does this work with all architectures?**
A: Best on Transformers (good spectral separation). May be weaker on CNNs or RNNs.

---

## Future Directions

1. **Real SVD integration**: Load actual model weights via safetensors
2. **OR-Tools ILP solver**: True integer linear programming (vs. greedy heuristics)
3. **QAT-aware metrics**: Fine-tuning effects on spectral structure
4. **GPU profiling**: Empirical VRAM/latency validation
5. **Ensemble methods**: Combine multiple quantization plans

---

**Last Updated**: 2026-06-14  
**License**: MIT
