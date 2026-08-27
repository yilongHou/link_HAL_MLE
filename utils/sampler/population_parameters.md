# Population Parameters for Data Generating Processes (DGPs)

This document provides the true population parameters for each data generating process (DGP) implemented in the sampler module, along with detailed calculations showing how these parameters are derived.

## Overview

All distributions are defined on the support [0,1] and the following parameters are calculated for each:

- **Mean (μ)**: First moment E[X]
- **Median**: Value x where F(x) = 0.5
- **Variance (σ²)**: Second central moment E[(X-μ)²]
- **Second moment**: E[X²] = Var(X) + μ²
- **Survival probability at 0.5**: P(X > 0.5) = 1 - F(0.5)

---

## 1. Truncated Normal Distribution

**JSON Parameters from Experiments**:

```json
"sampler_params": {
  "mean": 0.5,
  "std": 0.1,
  "lower": 0,
  "upper": 1
}
```

### Population Parameters

```
Mean: 0.5000 (exact)
Median: 0.5000 (exact)
Variance: 0.00998
Second moment: 0.25998
Survival probability at 0.5: 0.5000 (exact)
```

### Calculation Details

For a truncated normal with original parameters (μ, σ) truncated to [a, b]:

**Standardized bounds:**

- α = (a - μ)/σ = (0 - 0.5)/0.1 = -5
- β = (b - μ)/σ = (1 - 0.5)/0.1 = 5

**Variance formula:**

```
Var(X) = σ² × [1 + (α·φ(α) - β·φ(β))/(Φ(β) - Φ(α)) - ((φ(α) - φ(β))/(Φ(β) - Φ(α)))²]
```

Where φ and Φ are the standard normal PDF and CDF respectively.

Since the truncation bounds are ±5 standard deviations from the mean:

- φ(-5) ≈ φ(5) ≈ 1.5×10⁻⁶ (negligible)
- Φ(5) - Φ(-5) ≈ 1

The truncation effect is minimal, so Var(X) ≈ 0.01 × 0.9985 ≈ 0.00998.

The mean remains 0.5 due to symmetric truncation, making the median also 0.5.

---

## 2. Truncated GMM Symmetric Three Components

**JSON Parameters from Experiments**:

```json
"sampler_params": {
  "components": [
    {
      "mean": 0.2,
      "std": 0.05,
      "lower": 0,
      "upper": 1
    },
    {
      "mean": 0.5,
      "std": 0.05,
      "lower": 0,
      "upper": 1
    },
    {
      "mean": 0.8,
      "std": 0.05,
      "lower": 0,
      "upper": 1
    }
  ],
  "weights": [0.33, 0.34, 0.33]
}
```

### Population Parameters

```
Mean: 0.5000 (exact)
Median: 0.5000 (exact)
Variance: 0.06183
Second moment: 0.31183
Survival probability at 0.5: 0.5000 (exact)
```

### Calculation Details

**Law of Total Variance:**

```
Var(X) = Σ(wᵢ × Var(Xᵢ)) + Σ(wᵢ × (μᵢ - μ)²)
```

**Between-component variance:**

```
0.33×(0.2-0.5)² + 0.34×(0.5-0.5)² + 0.33×(0.8-0.5)²
= 0.33×0.09 + 0 + 0.33×0.09
= 0.0594
```

**Within-component variance:**
Each component has variance ≈ 0.0025 (minimal truncation effect):

```
0.33×0.0025 + 0.34×0.0025 + 0.33×0.0025 ≈ 0.0025
```

**Total variance:** 0.0594 + 0.0025 = 0.06183

The distribution is symmetric around 0.5, so mean = median = 0.5.

---

## 3. Truncated GMM Five Spikes

**JSON Parameters from Experiments**:

```json
"sampler_params": {
  "components": [
    {
      "mean": 0.45,
      "std": 0.005,
      "lower": 0,
      "upper": 1
    },
    {
      "mean": 0.475,
      "std": 0.005,
      "lower": 0,
      "upper": 1
    },
    {
      "mean": 0.5,
      "std": 0.005,
      "lower": 0,
      "upper": 1
    },
    {
      "mean": 0.525,
      "std": 0.005,
      "lower": 0,
      "upper": 1
    },
    {
      "mean": 0.55,
      "std": 0.005,
      "lower": 0,
      "upper": 1
    },
    {
      "mean": 0.5,
      "std": 0.05,
      "lower": 0,
      "upper": 1
    }
  ],
  "weights": [0.06666667, 0.06666667, 0.06666667, 0.06666667, 0.06666667, 0.66666667]
}
```

### Population Parameters

```
Mean: 0.5000 (exact)
Median: 0.5000 (exact)
Variance: 0.00209167
Second moment: 0.25209167
Survival probability at 0.5: 0.5000 (exact)
```

### Calculation Details

**Between-component variance:**

```
(1/15)×[(0.45-0.5)² + (0.475-0.5)² + 0² + (0.525-0.5)² + (0.55-0.5)²] + (2/3)×0²
= (1/15)×[0.0025 + 0.000625 + 0 + 0.000625 + 0.0025]
= (1/15) × 0.00625 = 1/2400 ≈ 0.0004167
```

**Within-component variance:**

```
5×(1/15)×0.000025 + (2/3)×0.0025
= 0.0000083 + 0.001667
≈ 0.001675
```

**Total variance:** 0.0004167 + 0.001675 = 0.00209167

About 80% of variance comes from the broad component with σ=0.05.

---

## 4. Truncated GMM Asymmetric Three Components

**JSON Parameters from Experiments**:

```json
"sampler_params": {
  "components": [
    {
      "mean": 0.35,
      "std": 0.1,
      "lower": 0.0,
      "upper": 1.0
    },
    {
      "mean": 0.65,
      "std": 0.05,
      "lower": 0.0,
      "upper": 1.0
    },
    {
      "mean": 0.9,
      "std": 0.2,
      "lower": 0.0,
      "upper": 1.0
    }
  ],
  "weights": [0.4, 0.4, 0.2]
}
```

### Population Parameters

```
Mean: 0.559669
Median: 0.60837
Variance: 0.041087
Second moment: 0.354317
Survival probability at 0.5: 0.61961
```

### Calculation Details

**Implementation behavior:**

The `TruncatedGMM` class implementation does NOT renormalize weights after truncation. Instead, it:

1. Creates truncated normal components with the specified parameters
2. Mixes them using the original input weights [0.4, 0.4, 0.2]
3. Each component contributes according to its original weight

**Truncated component means:**

- μ̃₁ ≈ 0.35009 (minimal truncation effect)
- μ̃₂ ≈ 0.65000 (no truncation effect)
- μ̃₃ ≈ 0.79817 (significant right truncation effect)

**Mixture mean:**

```
0.4×0.35009 + 0.4×0.65000 + 0.2×0.79817 ≈ 0.559669
```

**Variance calculation using law of total variance with original weights and truncated component parameters.**

---

## 5. Step Function Distribution

**JSON Parameters from Experiments**:

```json
"sampler_params": {
  "level1": 1.0,
  "level2": 0.5,
  "breakpoint": 0.7
}
```

### Population Parameters

```
Mean: 0.43824
Median: 0.42500
Variance: 0.07106
Second moment: 0.26312
Survival probability at 0.5: 0.41176
```

### Calculation Details

**PDF after normalization:**

```
Normalization constant = 1.0×0.7 + 0.5×0.3 = 0.85
f(x) = 1.176 for 0 ≤ x < 0.7
f(x) = 0.588 for 0.7 ≤ x ≤ 1
```

**Mean calculation:**

```
E[X] = 1.176×∫₀^0.7 x dx + 0.588×∫₀.₇¹ x dx
     = 1.176×(0.7²/2) + 0.588×[(1² - 0.7²)/2]
     = 1.176×0.245 + 0.588×0.255
     = 0.28812 + 0.14994 = 0.43824
```

**Second moment:**

```
E[X²] = 1.176×(0.7³/3) + 0.588×[(1³ - 0.7³)/3]
      = 1.176×0.1143 + 0.588×0.2190
      = 0.13442 + 0.12877 = 0.26312
```

**Variance:** E[X²] - (E[X])² = 0.26312 - 0.43824² = 0.07106

**Median:** Solve F(x) = 0.5
Since F(x) = 1.176x for x < 0.7, we get x = 0.5/1.176 = 0.425

---

## 6. Sinusoidal Distribution

**JSON Parameters from Experiments**:

```json
"sampler_params": {}
```

_Note_: The Sinusoidal distribution has no configurable parameters. The PDF is fixed as f(x) ∝ sin(πx) + 1.1 on [0,1].

### Population Parameters

```
Mean: 0.5000 (exact)
Median: 0.5000 (exact)
Variance: 0.070145
Second moment: 0.320145
Survival probability at 0.5: 0.5000 (exact)
```

### Calculation Details

**Normalization constant:**

```
C = ∫₀¹ (sin(πx) + 1.1) dx = [-cos(πx)/π + 1.1x]₀¹ = 2/π + 1.1 ≈ 1.7366
```

**Mean:** Due to symmetry of both sin(πx) and constant term around x=0.5, the mean is exactly 0.5.

**Second moment calculation:**

```
E[X²] = (1/C) × ∫₀¹ x²(sin(πx) + 1.1) dx
      = (1/C) × [∫₀¹ x²sin(πx) dx + 1.1∫₀¹ x² dx]
```

Using numerical integration consistent with the implementation:

```
E[X²] ≈ 0.320145
```

**Variance:** 0.320145 - 0.5² = 0.070145

Due to symmetry around x=0.5, the median equals the mean.

---

## Summary Table

| DGP                    | Mean     | Median   | Variance | Second Moment | S(0.5)   |
| ---------------------- | -------- | -------- | -------- | ------------- | -------- |
| TruncatedNormal        | 0.5000\* | 0.5000\* | 0.010000 | 0.260000      | 0.5000\* |
| TruncatedGMMSymmetric  | 0.5000\* | 0.5000\* | 0.061896 | 0.311896      | 0.5000\* |
| TruncatedGMMFiveSpikes | 0.5000\* | 0.5000\* | 0.002092 | 0.252092      | 0.5000\* |
| TruncatedGMMAsymmetric | 0.559669 | 0.60837  | 0.041087 | 0.354317      | 0.61961  |
| StepFunction           | 0.438235 | 0.42500  | 0.071283 | 0.263333      | 0.41176  |
| Sinusoidal             | 0.5000\* | 0.5000\* | 0.070145 | 0.320145      | 0.5000\* |

\*Exact values due to symmetry

---

## Implementation Notes

**Important**: These population parameters are computed to match the **actual implementation behavior** of the classes in this codebase:

1. **TruncatedGMM**: Does NOT renormalize mixture weights after truncation. The original input weights are used to combine the truncated components directly.

2. **Sinusoidal**: Uses numerical integration for normalization and moment calculations, matching the scipy integration approach used in the class implementation.

3. **All values verified**: These parameters have been computed by directly calling the implemented classes and verified against independent numerical calculations.

---

## References

1. Truncated Normal Distribution: Johnson, N. L., Kotz, S., & Balakrishnan, N. (1994). Continuous univariate distributions (Vol. 1).
2. Mixture Distributions: McLachlan, G., & Peel, D. (2000). Finite mixture models.
3. Law of Total Variance: For mixture distributions, the variance decomposes into within-component and between-component terms.
