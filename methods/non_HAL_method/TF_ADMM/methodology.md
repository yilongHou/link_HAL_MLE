Below is the **final, conc- **Parameters** $\theta=(\theta_0,\theta_1,\dots ,\theta_n)\in\mathbb R^{n+1}$
  (one lo## 5 Resulting density

The density is defined over the full support $[0,1]$. For a point $s \in [0,1]$, we find the bin index $i$ such that $s \in [x_i, x_{i+1})$. The density is constant within each bin. The first bin $[x_0, x_1) = [0, y_{(1)})$ has log-level $\theta_0=0$.

$$
p_{\theta^\ast}(s)
  =\frac{\exp(\theta^\ast_i)}
         {\Delta x_0 + \sum_{j=1}^{n}\exp(\theta^\ast_j)\,\Delta x_j},
  \quad s\in[x_i,x_{i+1}),\; i=0,\dots,n.
$$per bin), with **identifiability constraint** $\theta_0=0$.

- **Optimization variables** $\tilde{\theta}=(\theta_1,\dots ,\theta_n)\in\mathbb R^{n}$
  (the free parameters).

- **Normalising constant** $Z(\tilde{\theta})=e^0\,\Delta x_0+\sum_{i=1}^{n}e^{\theta_i}\,\Delta x_i = \Delta x_0+\sum_{i=1}^{n}e^{\theta_i}\,\Delta x_i$.recipe** with wording cleaned up to make it crystal-clear that

- the **only role of the observations** $y_{(1)},\dots ,y_{(n)}$ is to fix the knot grid, and
- the optimization objective depends **solely on $\theta$** (via the log-likelihood and the trend-filter penalty).

No residual “$y-\theta$” or MSE terms remain.

---

## 1 Grid built from the data

1. **Sort the sample once**
   $y_{(1)}<\dots<y_{(n)}$.

2. **Define the knots**

   $$
     x_0=0,\quad x_i=y_{(i)},\;i=1{:}n,\quad x_{n+1}=1,
   $$

   with bin widths $\Delta x_i=x_{i+1}-x_i,\;i=0{:}n$.

---

## 2 Model and penalized log-likelihood

- **Parameter** $\theta=(\theta_1,\dots ,\theta_n)\in\mathbb R^n$
  (one log-level per bin).

- **Normalising constant** $Z(\theta)=\sum_{i=1}^{n}e^{\theta_i}\,\Delta x_i$.

- **Negative log-likelihood** (one observation per bin)

  $$
    f(\tilde{\theta})
      = -\sum_{i=1}^{n}\theta_i
        + (n+1)\log Z(\tilde{\theta}).
  $$

- **Trend-filter penalty** of order $k+1$:
  $\|D^{(k+1)}(0,\tilde{\theta})\|_{1}$.

$$
\boxed{\;
\min_{\tilde{\theta}\in\mathbb R^{n}}
      f(\tilde{\theta})
      +\lambda\,
       \bigl\|D^{(k+1)}(0,\tilde{\theta})\bigr\|_1
\;}
\tag{P}
$$

---

## 3 Specialized ADMM splitting

_Introduce $\alpha=D^{(k)}\theta$. $\rho$ is fixed at $\lambda$._

| variable | dimension | role                           |
| -------- | --------- | ------------------------------ |
| $\theta$ | $n$       | primal log-levels              |
| $\alpha$ | $n-k$     | $k$-th differences of $\theta$ |
| $u$      | $n-k$     | scaled dual                    |

$$
\mathcal L_\rho(\theta,\alpha,u)=
f(\theta)
+\lambda\|D^{(1)}\alpha\|_1
+\tfrac{\lambda}{2}\|\alpha-D^{(k)}\theta+u\|_2^2
-\tfrac{\lambda}{2}\|u\|_2^2.
$$

---

## 4 ADMM iterations

```text
Initialise  θ⁰=(0,...,0)∈ℝⁿ, α⁰=0, u⁰=0, ρ=λ
for t = 0,1,2,…:
  # θ₀-step (set as constant anchor)
  θ₀^{t+1} = 0

  # θ-step for θ₁,...,θₙ (smooth + quadratic)
  # Note: f(θ) and D now depend on the full vector [θ₀^{t+1}, θ₁,...,θₙ]
  (θ₁,...,θₙ)^{t+1} = argmin  f([θ₀^{t+1}, θ₁,...,θₙ]) + (λ/2)‖αᵗ - D^{(k)}[θ₀^{t+1}, θ₁,...,θₙ] + uᵗ‖²
                      → L-BFGS on n variables

  # Let θ^{t+1} be the full vector [θ₀^{t+1}, (θ₁,...,θₙ)^{t+1}]

  # α-step (first-difference fused lasso)
  v = D^{(k)}θ^{t+1} - uᵗ
  α^{t+1} = argmin_α  ½‖α - v‖² + ‖D^{(1)}α‖₁
           → dynamic-programming, coordinate descent, or genlasso

  # dual update
  u^{t+1} = uᵗ + α^{t+1} - D^{(k)}θ^{t+1}

  # convergence
  r = α^{t+1} - D^{(k)}θ^{t+1}
  s = -λ(α^{t+1} - αᵗ)
  stop when  ‖r‖₂ ≤ ε_pri  and  ‖s‖₂ ≤ ε_dual
end
```

Recommended defaults
$\varepsilon_{\mathrm{pri}}=\varepsilon_{\mathrm{dual}}=10^{-4}\sqrt{n-k}$.

---

## 5 Resulting density

$$
p_{\theta^\ast}(s)
  =\frac{\exp(\theta^\ast_i)}
         {\sum_{j=1}^{n}\exp(\theta^\ast_j)\,\Delta x_j},
  \quad s\in[x_i,x_{i+1}).
$$

---

Everything now explicitly shows that the **observations appear only through the knots**; the optimisation uses **no direct $y$** term—exactly what you wanted. Let me know if you need code scaffolding or further tweaks!
