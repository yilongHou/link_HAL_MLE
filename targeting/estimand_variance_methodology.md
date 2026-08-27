# Estimand Variance Methodology

This document outlines the methodology for variance estimation in targeted maximum likelihood estimation (TMLE) for various statistical functionals.

## Survival Function and CDF

For the marginal survival function at $x_0$:

$$D^*_f(x) = I(x > x_0) - \mathbb{E}_{P_f}[I(x > x_0)]$$

Thus, $h = I(x > x_0)$.

Similarly, for the cumulative distribution function (CDF):

$$h = I(x < x_0)$$

## Median

For the median estimator:

$$D^*_f(x) = \frac{\frac{1}{2} - I(x < \tilde{X})}{f(\tilde{X})}$$

where $\tilde{X}$ is the true median, and thus $h = I(X < \hat{F}^{-1}(0.5))$.

**Implementation Note:** We need to estimate the median based on the initial estimation and then perform the targeting step. This formula for the median can be generalized to any percentile.

## K-th Order Moments

For k-th order moments:

$$D^*_f(x) = x^k - \mu^k$$

where $\mu$ is the true mean, and thus $h = x^k$.

**Note:** This represents the parametric basis. One explanation for not penalizing the parametric part (similar to TF and LAS approaches) is that we estimate the density while targeting all its k-th order moments.

## Variance Estimation

TMLE is asymptotically linear with influence curve $D^*_f$. Therefore, we can simply take the sample variance of $D^*_f$ as the variance estimation for $\Phi^F(f_n)$:

$$\text{Var}(\Phi^F(f_n)) \approx \frac{1}{n} \text{Var}(D^*_f)$$

where the sample variance provides a consistent estimator of the asymptotic variance.
