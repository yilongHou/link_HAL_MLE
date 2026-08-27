Below is a practical guideline from “choose (k)” through “write the objective,” in a form that maps cleanly to a CVXPY implementation. I will keep it at the level where you can implement directly, but without committing to a particular indexing convention beyond what you already use.

---

## A. Choose the modeling objects

### A1) Pick the trend-filtering order (k)

- (k=0): piecewise-constant (TV / fused-lasso on (\theta)).
- (k=1): piecewise-linear (penalize second differences).
- In general: penalize ((k+1))st differences; the “parametric part” has dimension (k+1).

### A2) Build the data-adaptive grid

Given (y_1,\dots,y_n\in[0,1]):

1. Sort: (y*{(1)}\le \cdots \le y*{(n)}).
2. Define knots: (x*0=0,; x_i=y*{(i)}\ (i=1,\dots,n),; x\_{n+1}=1).
3. Widths: (\Delta x*i=x*{i+1}-x_i,\ i=0,\dots,n).

Let (m:=n+1) and parameterize
[
\theta=(\theta_0,\dots,\theta_n)\in\mathbb R^{m}.
]
Your log-density is constant on each bin ([x*i,x*{i+1})).

### A3) Compute bin counts (c_i)

Define
[
c*i := \sum*{j=1}^n \mathbf 1{y*j\in[x_i,x*{i+1})},\qquad i=0,\dots,n.
]
In your “knot at each sample” construction, typically (c_i=1) for (i=1,\dots,n) and (c_0=0), but computing (c) is safer (ties, convention, etc.).

### A4) Fix identifiability (mandatory)

Because (p\_\theta) is invariant to (\theta\mapsto\theta + c\mathbf 1), impose one constraint, e.g.

- **anchor**: (\theta_0=0) (simplest for CVXPY), or
- **centering**: (\sum\_{i=0}^n \Delta x_i,\theta_i=0).

I recommend (\theta_0=0) for implementation clarity.

---

## B. Define the likelihood term (convex)

### B1) Discrete normalizing constant

[
Z(\theta) = \sum_{i=0}^{n} \exp(\theta_i),\Delta x_i.
]

### B2) Negative log-likelihood

Using the count form:
[
f(\theta)
= -\sum\_{i=0}^{n} c_i\theta_i

- n\log!\Big(\sum\_{i=0}^{n} e^{\theta_i}\Delta x_i\Big).
  ]

**CVXPY note:** implement the log-sum-exp stably as
[
\log\Big(\sum_i \Delta x_i e^{\theta_i}\Big)
============================================

\mathrm{logsumexp}(\theta + \log \Delta x),
]
where (\log \Delta x) is taken elementwise.

So in code form: `cp.log_sum_exp(theta + log_dx)`.

---

## C. Build the penalty operator that includes the parametric part

You want to penalize the full falling-factorial coefficient vector
[
\alpha = (H^{(k)})^{-1}\theta
=============================

\begin{bmatrix}
C[1mm]
\frac{1}{k!}D^{(k+1)}
\end{bmatrix}\theta,
]
and use (|\alpha|\_1).

So you need to construct:

### C1) The divided-difference / discrete derivative matrices (D^{(i)}) and (\Delta^{(i)})

For irregular spacing (your adaptive grid), these are not the simple equally-spaced finite-difference matrices; they are the “discrete derivative” operators used in falling-factorial / trend-filtering with arbitrary inputs.

Concretely, you need:

- (D^{(1)},\dots,D^{(k+1)}),
- (\Delta^{(1)},\dots,\Delta^{(k)}) (diagonal scaling matrices built from spacings),
  constructed by the same recursion you already referenced (Ramdas et al. / TF literature) but evaluated on your grid.

Dimensions (typical):

- (\theta\in\mathbb R^{m}),
- (D^{(i)}\theta\in\mathbb R^{m-i}).

(Your current code already builds (D^{(k+1)}) for the TF penalty; reuse that.)

### C2) Construct (C\in\mathbb R^{(k+1)\times m})

Use:

- (C_1 = e_1^\top),
- For (i=1,\dots,k),
  [
  C_{i+1}
  =
  \Big(\tfrac{1}{(i-1)!}(\Delta^{(i)})^{-1}D^{(i)}\Big)_{1,\cdot}.
  ]
  This gives you the “parametric part” rows that correspond to the first (k+1) falling-factorial coefficients.

### C3) Form the full inverse-basis operator (H\_{\text{inv}})

Define
[
H*{\text{inv}}^{(k)}
:=
\begin{bmatrix}
C[1mm]
\frac{1}{k!}D^{(k+1)}
\end{bmatrix}\in\mathbb R^{m\times m}.
]
Then the full coefficient vector is (\alpha = H*{\text{inv}}^{(k)}\theta), and your penalty is (|H\_{\text{inv}}^{(k)}\theta|\_1).

---

## D. Final optimization objective (ready for CVXPY)

Let (\theta\in\mathbb R^{m}) be the decision variable and impose (\theta_0=0). The full-penalty TF density estimator solves:

[
\boxed{
\min_{\theta\in\mathbb R^{m}}
\left{
-\sum_{i=0}^{n} c_i\theta_i
;+;
n\log!\Big(\sum_{i=0}^{n} e^{\theta_i}\Delta x_i\Big)
;+;
\lambda\big|H_{\text{inv}}^{(k)}\theta\big|_1
\right}
\quad \text{s.t. }\theta_0=0.
}
]

**CVXPY form (symbolically):**

- `theta = cp.Variable(m)`
- `logZ = cp.log_sum_exp(theta + log_dx)` (where `log_dx = np.log(dx)`)
- `nll = -c @ theta + n * logZ`
- `pen = lam * cp.norm1(Hinv @ theta)`
- `constraints = [theta[0] == 0]`
- `prob = cp.Problem(cp.Minimize(nll + pen), constraints)`

This is convex: `log_sum_exp` is convex, `norm1` is convex, and you are minimizing a convex objective with linear constraints.

---

## Implementation cautions (so CVXPY behaves)

1. **Don’t drop (\theta_0)** if you penalize the parametric part—keep the full (\theta) length (m) and just constrain (\theta_0=0). Otherwise (C\theta) is ill-defined / inconsistent.

2. **Use log-sum-exp with (\log \Delta x)** (avoid explicit `exp` and `sum`).

3. **Scaling:** (\lambda) now hits ((k+1)) extra coefficients, so the effective regularization is stronger than classic TF. In practice you may want a different scaling for the parametric rows (e.g., weights), but the “correct” full (\ell*1) coefficient norm is exactly (|H*{\text{inv}}\theta|\_1).

---

If you paste (or summarize) your current construction of (D^{(i)}) and (\Delta^{(i)}) (shapes and recursion), I can write the explicit “matrix assembly” steps for (C) and (H\_{\text{inv}}) in the same notation as your appendix, including exact dimensions for each block, so you can drop it straight into the paper.
