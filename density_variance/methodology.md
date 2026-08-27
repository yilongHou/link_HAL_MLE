Some methodology:

\subsubsection{Covariance Estimation for $\beta_n$} \label{covariance*beta}
The covariance matrix $\operatorname{Cov}(\hat{\beta})$ is typically estimated by inverting the observed Fisher information matrix. Denote by $S(O_i;\beta)=\frac{\partial L*{\beta}(O*i)}{\partial \beta}$ the contribution to the score (i.e., the gradient of log-likelihood) from an observation $O_i$. Then the observed information is estimated as
\[
\hat{I}(\hat{\beta}) = \frac{1}{n}\sum*{i=1}^{n} S(O*i;\hat{\beta})\,S(O_i;\hat{\beta})^\top.
\]
In addition, we add a ridge regularization term in case $I(\hat{\beta})$ is nearly singular:
\[
\hat{I}*{\text{reg}}(\hat{\beta}) = \hat{I}(\hat{\beta}) + \lambda I,
\]
with $\lambda>0$ a small constant and $I$ the identity matrix. Then,
\[
\hat{\operatorname{Cov}}(\hat{\beta}) = \frac{1}{n}\,\hat{I}\_{\text{reg}}(\hat{\beta})^{-1}.
\]

\subsubsection{Confidence Interval for the Density $f(x_0;\beta)$ at $x_0$}
For a fixed point $x_0$, the density is
\[
f(x*0;\beta) = \frac{\exp\bigl\{\theta_0 + \phi(x_0)^\top \beta\bigr\}}{C(\beta)}.
\]
Treating $f(x_0;\beta)$ as a function of $\beta$ (denoted $f(\beta;x_0)$), its gradient with respect to $\beta$ is
\[
\nabla*\beta f(x*0;\beta) = f(x_0;\beta)\,\Bigl[\phi(x_0) - E\{\phi(X)\}\Bigr],
\]
where
\[
E\{\phi(X)\} = \int \phi(x)\,f(x;\beta)\,dx.
\]
By the delta method, the variance of $f(x_0;\hat{\beta})$ is
\[
\operatorname{Var}\{f(x_0;\hat{\beta})\} = f(x_0;\hat{\beta})^2\,
\Bigl[\phi(x_0) - E\{\phi(X)\}\Bigr]^\top
\operatorname{Cov}(\hat{\beta})\,
\Bigl[\phi(x_0) - E\{\phi(X)\}\Bigr].
\]
Thus, the final $100(1-\alpha)\%$ confidence interval for $f(x_0;\beta)$ is:
\begin{align}
\hat{f}(x_0;\hat{\beta}) \; \pm \; & z*{1-\alpha/2}\,\hat{f}(x_0;\hat{\beta})\,
\sqrt{
\Bigl[\phi(x_0) - E\{\phi(X)\}\Bigr]^\top
\operatorname{Cov}(\hat{\beta})
\Bigl[\phi(x_0) - E\{\phi(X)\}\Bigr]
}.
\label{eq:ci_density}
\end{align}
The covariance matrix $\operatorname{Cov}(\hat{\beta})$ is estimated as described above.

We evaluate the finite‑sample performance of this variance estimator in Section~\ref{Simulation}.

Some draft code for the methodology section of the paper, focusing on the estimation of the variance of the density estimate using the delta method and confidence intervals.
