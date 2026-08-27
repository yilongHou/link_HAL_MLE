Below is a comprehensive, step-by-step guide for defining several DGPs on $[0,1]$, implementing their PDF, sampling routine, and CDF, and computing their variational norms (VN) for orders $k=0,1,2,3$. The step function only uses $k=0$.

---

## 1. Setup

```python
import numpy as np
from scipy.stats import truncnorm, norm
from scipy.integrate import quad
from scipy.interpolate import interp1d
```

---

## 2. DGP Definitions

### 2.1 Truncated Normal on $[0,1]$

```python
def truncated_normal_pdf(x, mu=0.5, sigma=0.1, a=0, b=1):
    a_std, b_std = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.pdf(x, a_std, b_std, loc=mu, scale=sigma)

def truncated_normal_sample(size, mu=0.5, sigma=0.1, a=0, b=1):
    a_std, b_std = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.rvs(a_std, b_std, loc=mu, scale=sigma, size=size)

def truncated_normal_cdf(x, mu=0.5, sigma=0.1, a=0, b=1):
    a_std, b_std = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.cdf(x, a_std, b_std, loc=mu, scale=sigma)
```

* **PDF**: standard `scipy.stats.truncnorm`.
* **Sampling**: direct `rvs` from truncated normal.
* **CDF**: `cdf` method of `truncnorm`.

---

### 2.2 Sinusoidal-based Density on $[0,1]$

Define $f(x) \propto \sin(\pi x) + 1.1$ to keep it strictly positive.

```python
def sinusoidal_pdf(x):
    # Normalizing constant
    norm_const, _ = quad(lambda t: np.sin(np.pi * t) + 1.1, 0, 1)
    return (np.sin(np.pi * x) + 1.1) / norm_const

def sinusoidal_cdf(x):
    # Integrate PDF from 0 to x
    integral, _ = quad(sinusoidal_pdf, 0, x)
    return integral

def sinusoidal_sample(size):
    # Precompute inverse CDF via interpolation
    grid = np.linspace(0, 1, 2000)
    cdf_vals = np.array([sinusoidal_cdf(xi) for xi in grid])
    inv_cdf = interp1d(cdf_vals, grid, kind='linear', fill_value=(0,1), bounds_error=False)
    u = np.random.uniform(0, 1, size=size)
    return inv_cdf(u)
```

* **PDF**: normalized $\sin(\pi x)+1.1$.
* **CDF**: numerical integration of the PDF.
* **Sampling**: inverse-transform using a dense grid and `interp1d`.

---

### 2.3 Truncated Gaussian Mixture Model (GMM) on $[0,1]$

Mixture of two normals with weights $p_1=0.6$, $p_2=0.4$. Truncate to $[0,1]$.

```python
def truncated_gmm_pdf(x):
    p1, mu1, sigma1 = 0.6, 0.3, 0.05
    p2, mu2, sigma2 = 0.4, 0.7, 0.07
    raw = p1 * norm.pdf(x, mu1, sigma1) + p2 * norm.pdf(x, mu2, sigma2)
    norm_const, _ = quad(lambda t: p1*norm.pdf(t, mu1, sigma1) + p2*norm.pdf(t, mu2, sigma2), 0, 1)
    return raw / norm_const

def truncated_gmm_cdf(x):
    integral, _ = quad(truncated_gmm_pdf, 0, x)
    return integral

def truncated_gmm_sample(size):
    p1, mu1, sigma1 = 0.6, 0.3, 0.05
    p2, mu2, sigma2 = 0.4, 0.7, 0.07

    # 1. Determine how many samples from component 1 vs. 2
    n1 = np.random.binomial(size, p1)
    n2 = size - n1

    # 2. Sample from each normal, then truncate to [0,1]
    #    We generate a slightly larger batch to reduce rejection loops
    extra_factor = 1.2
    samples1 = norm.rvs(mu1, sigma1, size=int(n1 * extra_factor))
    samples2 = norm.rvs(mu2, sigma2, size=int(n2 * extra_factor))

    samples1 = samples1[(samples1 >= 0) & (samples1 <= 1)][:n1]
    samples2 = samples2[(samples2 >= 0) & (samples2 <= 1)][:n2]

    # 3. If too few due to truncation, keep resampling until counts are met
    while len(samples1) < n1:
        extra = norm.rvs(mu1, sigma1, size=(n1 - len(samples1)))
        extra = extra[(extra >= 0) & (extra <= 1)]
        samples1 = np.concatenate([samples1, extra])
    while len(samples2) < n2:
        extra = norm.rvs(mu2, sigma2, size=(n2 - len(samples2)))
        extra = extra[(extra >= 0) & (extra <= 1)]
        samples2 = np.concatenate([samples2, extra])

    return np.concatenate([samples1[:n1], samples2[:n2]])
```

* **PDF**: mixture of two normals renormalized over $[0,1]$.
* **CDF**: integrate the truncated GMM PDF.
* **Sampling**:

  1. Draw $n_1 \sim \operatorname{Binomial}(size,\,p_1)$, $n_2 = size - n_1$.
  2. Generate slightly more than $n_1$ and $n_2$ from each normal, truncate to $[0,1]$.
  3. If the truncated batch is too small, resample until you have exactly $n_1$ and $n_2$.
  4. Concatenate.

---

### 2.4 Step Function on $[0,1]$

Define

$$
f(x) \;=\;
\begin{cases}
1.0, & x < 0.5,\\
0.5, & x \ge 0.5.
\end{cases}
$$

```python
def step_function_pdf(x):
    return 1.0 if x < 0.5 else 0.5

def step_function_cdf(x):
    if x < 0:
        return 0.0
    elif x < 0.5:
        # area = ∫_0^x 1.0 dx = x
        return x
    elif x <= 1:
        # area up to 0.5 is 0.5*1.0 = 0.5
        # plus ∫_{0.5}^x 0.5 dx = 0.5 * (x - 0.5)
        return 0.5 + 0.5 * (x - 0.5)
    else:
        return 1.0

def step_function_sample(size):
    # We generate U ~ Uniform(0,1). Then solve for X s.t. CDF(X) = U.
    u = np.random.uniform(0, 1, size=size)
    samples = np.where(u < 0.5,
                       u / 1.0,                    # if u < 0.5, x = u
                       0.5 + (u - 0.5) / 0.5 * 0.5 # if u >= 0.5, invert piecewise
                      )
    return samples
```

* **PDF**: two constant levels.
* **CDF**: piecewise linear.
* **Sampling**: direct inversion of CDF.

