\section{Introduction} \label{intro}
Let $T$ be a non-negative random variable with absolutely continuous distribution function $F_T$, survival function $S = 1 - F_T$, density $p_T$, and hazard rate $\alpha = p_T/S$. WLOG, let the support of $F$ be $[0,1]$. A standard approach for estimating $S = P(T > t)$ is by estimating $\alpha$
\citep{andersen2012statistical, gill1990survey, aalen1978empirical, rytgaard2023estimation}.
However, the unbounded tail behavior of the hazard function makes the estimation unreliable. Various boundary‑correction schemes have been proposed—e.g., kernel‑smoothing of the Nelson–Aalen increments \citep{ramlau1983smoothing}, spline‑based tail adjustments \citep{wang2005smoothing}—but these methods often rely on heuristic tuning without a unified theoretical framework. Another reason for not estimating the hazard is included in Appendix A.3 of \citet{kooperberg1992logspline}. In our opinion, the tail behavior is caused by the denominator, $S(t)$, going to 0 when $t$ goes to 1. Thus, it is more reasonable to believe the density of failure is well-behaved, such as the simple uniform distribution. With the density estimated, one can easily recover the marginal survival probability by integration.

For the density estimation of $T$, a prominent existing approach is the logspline density estimator \citep{kooperberg1992logspline, kooperberg1991study}. Logspline models approximate the log-likelihood by a linear combination of cubic spline or B-spline basis functions, with coefficients estimated via maximum likelihood estimation. To handle censored observations, a few cycles of the steepest descent algorithm and an extra Newton-Raphson algorithm are adopted \citep{kooperberg1992logspline}. Although highly flexible, they have three major problems. Firstly, logspline estimation can produce pronounced oscillations or “spikes” in the fitted density when norm or shape constraints are not enforced, leading to instability in finite samples. Secondly, in Appendix A.3 of \citep{kooperberg1992logspline}, it is stated that there is no guarantee that their estimates are the global minimum of the log-likelihood function, due to the non-log-concavity nature of the likelihood function for the coarsened data. Lastly, their method has no generalizability to higher dimensions.

In this paper, we develop a family of univariate density estimators based on the Highly Adaptive Lasso (HAL) \cite{benkeser2016highly}, which exhibits robust finite-sample performance, possesses a theoretical guarantee of uniqueness of the solution, and is generalizable to the multivariate setting while maintaining strong theoretical guarantees. We establish this by first demonstrating the relationship between univariate HAL regression and existing total variation (TV) denoising spline regression methods (i.e., restricted and unrestricted LAS \citep{mammen1997locally} and trend filtering \citep{tibshirani2014adaptive}). Then, we show the optimal $L_2$ convergence rate, uniform convergence, and pointwise asymptotic normality of HAL-MLE. For handling coarsened data under the coarsening at random (CAR) assumption \cite{gill1997coarsening}, we embed HAL‑MLE within the EM algorithm (EM‑HAL‑MLE) to recover a pseudo-full-data structure, which allows us to deal with coarsened data just like full data under the CAR assumption. We propose the variance estimations through the Delta method and the influence curve of the marginal survival function. Finally, we illustrate these with simulations and a case study.

\section{Related Work} \label{lit review}

In this section, we proceed as follows. First, we introduce Highly Adaptive Lasso (HAL) regression and establish its connection to existing total‐variation denoising spline methods via the representation theorem for \emph{càdlàg} functions with bounded sectional variation. Next, we formally define a family of HAL‐MLE estimators and analyze their convergence properties and asymptotic behavior. We then present the nonparametric maximum likelihood estimation (NPMLE) perspective on HAL‐MLE alongside classical NPMLE approaches, such as the Kaplan–Meier estimator. Finally, we review the targeted maximum likelihood estimation (TMLE) framework and the EM algorithm, highlighting their roles in survival analysis.

\subsection{The Highly adaptive lasso (HAL) Estimator} \label{HAL Estimator}
The HAL estimator \citep{benkeser2016highly} is proposed for modeling the behavior of a subclass of the \emph{càdlàg} functions on $[0,1]^d$ with bounded variation. A stronger norm, called the sectional variational norm, is introduced such that its Donsker property \citep{van1996weak} can be extended to general dimension $d$.

\paragraph{Basic Assumptions.} \label{Basic Assumptions}
We start by introducing the function class of \emph{c\`adl\`ag} functions with bounded sectional variation norm (upper bounded by \(M\)) without assuming any order of differentiability, denoted as $D^{(0)}_M\bigl([0,1]^d\bigr).$

We note that any \(f\in D^{(0)}_M\bigl([0,1]^d\bigr).\) can be represented as
\[
f(x)
= f(0) + \sum_{s\subset\{1,\dots,d\}}
\int*{(0(s),\,x(s)]} f\bigl(du_s,0(-s)\bigr),
\]
where \(s\) is any nonempty subset of \(\{1,\dots,d\}\), and \(0(-s)\) denotes the vector in which coordinates not in \(s\) are set to zero. Accordingly, the sectional variation norm is defined by
\[
\|f\|\_v^\*
= \bigl|f(0)\bigr| + \sum*{s\subset\{1,\dots,d\}}
\int\_{(0(s),\,x(s)]}
\Bigl|\,f\bigl(du(s),0(-s)\bigr)\Bigr|.
\]
This is a stronger assumption than the bounded total variation (BTV) assumption, since it requires bounded variation not only on the full coordinate set \(s=\{1,\dots,d\}\) but on all lower‑dimensional sections \(s\subset\{1,\dots,d\}\).

In the univariate case $D^{(0)}_M\bigl([0,1]\bigr)$, these simplify to
\begin{align}
f(x)
= f(0) + \int*{(0,x]} f(du),
\quad
\|f\|\_v^\*
= \bigl|f(0)\bigr| + \int*{(0,x]}
\Bigl|\,f(du)\Bigr|.
\end{align}

\textbf{\textit{Remark:}} In the univariate case, a \emph{c\`adl\`ag} function could be considered as a linear combination of the CDFs. For more on \emph{c\`adl\`ag} functions with bounded sectional variation norm and their measurability, one could refer to Section 2 of \citep{munch2024estimating}.

\paragraph{Representation for k-th order differentiable $f$} \label{Representation for k-th order differentiable}

Throughout this section we assume \(f\in D^{(k)}\_M\bigl([0,1]\bigr)\).

\begin{assumption}[The k-th Order Smoothness Class]\label{assump:DkM}
Let the order $k\ge1$ and a large enough constant $M>0$. We say $f\in D^{(k)}_M\bigl([0,1]\bigr)$ if the following hold:
\begin{enumerate}[label=(\roman*)]
\item $f\in D^{(0)}_M\bigl([0,1]\bigr)$, i.e.\ $f$ is \emph{c\`adl\`ag} on $[0,1]$.
\item For each $1\le i\le k$, the $i$‑th Lebesgue–Radon–Nikodym derivative exists
\[
f^{(i)}(u)\;=\;\frac{d}{du}f^{(i-1)}(u)
\]
and $f^{(i)}\in D^{(0)}_M\bigl([0,1]\bigr)$.
\item The $k$‑th order sectional variation norm of $f$,
defined by
\[
\|f\|_{v,k}^\*
\;=\sum_{i=0}^k |f^{(k)}(0)| + \|f^{(k)}\|_v,
\]
satisfies
\[
\|f\|_{v,k}^\* \;\le\; M.
\]
\end{enumerate}
\end{assumption}

We then write $(x - u)_+^k = (x - u)^k\,I\{u\le x\}$ where $I$ is the indicator, and this truncated $k$-th order spline with knot point $u \in [0,1]$ is $k$-th order weakly differentiable. Notice that the following representation theorem and its derivation are specialized to the univariate case; for the multivariate or higher‑order spline representation, please check out Section 3 of \citep{vanderlaan2023higherordersplinehighly}.

\textbf{\textit{Remark:}} For concreteness, an example of the representation and basis system for the multivariate case is listed in the Appendix \ref{appendix:A2}. One can fully understand the material without needing to read this Appendix.

\begin{theorem}[Representation for \(f\in D^{(k)}_M(\lbrack0,1\rbrack)\)]
\label{thm:HAL_representation}

Under the above assumptions, any \(f\in D^{(k)}_M\bigl([0,1]\bigr)\) could be represented as follows:
\[
f(x)
= \sum_{i=0}^k \frac{1}{i!}\,f^{(i)}(0)\,x^i + \int*0^1 \frac{1}{k!}\,(x-u_k)*+^k \,d\,f^{(k)}(u_k).
\]
\end{theorem}

\begin{proof}
See Appendix \ref{appendix:A.1}
\end{proof}

\paragraph{Estimation Construction.} \label{Estimation Construction}
Based on the representation theorem \ref{thm:HAL_representation}, we could construct our estimation with a finite-dimensional working model by approximating the integral with a sum. Suppose we partition the integral into $J$ parts with knot points $x_1, \cdots, x_J$. Let \(f_0\in D^{(k)}\_M([0,1]^d)\) be the truth, and \(f_J\) be the estimation based on the $J$-knot points. Note that the dimension of the working model here refers to the parameters needed for modeling rather than the structure of observations.

\[
f*J(x)
= \sum*{i=0}^k \frac{f^{(i)}(0)}{i!}\,x^i + \sum*{j =1}^J \frac{\Delta\,f^{(k)}(x_j)}{k!}\,(x-x_j)*+^k \,.
\]

And we could summarize the basis system as follows:

\[
\phi*{i,0}(x) = x^i, \quad i = 0, \ldots, k,
\]
\[
\phi*{k,x*j}(x) = (x - x_j)*+^k, \quad j = 1, \ldots, J.
\] \label{basis_d1}

This basis naturally divides into:
\begin{itemize}[leftmargin=*]
\item The \textbf{parametric part}: the global polynomial trend ($x^i$-type terms),
\item The \textbf{nonparametric part}: the truncated power functions that allow flexible deviations at specified knots $x_{1:n}$.
\end{itemize}

\textbf{\textit{Remark:}} Such a basis system is also called the truncated power basis (Theorem 8.51 of \citet{schumaker2007spline}).

In practice, we estimate \(f*0\) by estimating the coefficients $\beta$ of the basis constructed by sample $x*{1:n}$.
The coefficient vector should be of the form \(\beta=(\beta*0,\dots,\beta*{k+n})\), the estimator is
\[
f*{n,\beta}(x)
= \sum*{i=0}^k \beta*i\,\phi*{i,0}(x) + \sum*{j=1}^n \beta*{k+j}\,\phi\_{k,x_j}(x).
\]

Hence, its $k$-th order sectional variation norm is

$ \bigl\|f*{n,\beta}\bigr\|*{v,k}^\* = \|\beta\|\_1.$

And for a general loss \(L\), the empirical risk minimizer is
\begin{align} \label{equation: HAL beta optimizer}
\beta*n(M)
= \arg\min*{\|\beta\|_1 \le M}
P_n\,L\bigl(f_{n,\beta}\bigr),
\qquad
\hat f(x)=f\_{n,\beta_n(M)}(x).  
\end{align}

\subsection{Unifying the Methods for Univariate Spline Regression with Variational Penalty} \label{Splines Methods}

HAL turns out to be the solution to a variational optimization problem, even though it was not its original intention. In this section, we will introduce some general methods in univariate spline regression with variational penalty. Then, we will demonstrate the strong relationship between these methods and univariate HAL from the perspectives of assumptions, theoretical properties, and implementation.

We warm up by discussing the major theoretical limitation of linear smoothers, which is the major improvement of local adaptive splines. These linear smoothers, whose fitted values depend linearly on the observed responses, include smoothing splines \citep{wahba1990spline}, kernel regression \citep{stone1982optimal}, and etc. These methods are only optimal, in terms of $L_2$ convergence, in Sobolev or Hölder spaces \citep{tsybakov2009nonparametric}, and suboptimal elsewhere in the function class with bounded total variation. For concreteness, when the true function is smooth in some parts of its domain and wiggly in other parts, these linear smoothers -- the estimators with the property for the fitted values to be a linear function of the responses -- become suboptimal \cite{tibshirani2014adaptive}.

\citet{mammen1997locally} proposed locally adaptive regression splines (LAS), which fixed this problem and achieved optimal rates in $\mathcal{F}_k$, where

\[
\mathcal{F}\_k
\;=\;
\bigl\{\,f : [0,1]\to\mathbb{R}
\;\big|\;
f\text{ is $k$-th order differentiable and }
\mathrm{TV}\bigl(f^{(k)}\bigr)<\infty
\bigr\}.
\]

The optimization over $\mathcal{F}_k$ is referred to as the unrestricted LAS method as follows:
\begin{equation}\label{eq:ula-spline}
\hat f \;\in\;\arg\min*{f\in\mathcal{F}\_k}
\frac{1}{2}\sum*{i=1}^n \bigl(y_i - f(x_i)\bigr)^2
\;+\;\lambda\,\mathrm{TV}\bigl(f^{(k)}\bigr),
\end{equation}

Due to the difficulty of implementation, especially when $k \ge 2$, they proposed the restricted LAS method as an asymptotic solution, as follows:

\begin{align*}
\mathcal{F}*{n,k}
&=
\bigl\{\,f:[0,1]\to\mathbb{R}
\;\big|\;
f\text{ is the linear combination of the $k$-th order splines with knots in }x*{1:n},\\
&\qquad\mathrm{TV}\bigl(f^{(k)}\bigr)<\infty
\bigr\},
\end{align*}

and the optimization problem becomes

\begin{equation}\label{eq:rla-spline}
\hat f \;\in\;\arg\min*{f\in\mathcal{F}*{n,k}}
\frac{1}{2}\sum\_{i=1}^n \bigl(y_i - f(x_i)\bigr)^2
\;+\;\lambda\,\mathrm{TV}\bigl(f^{(k)}\bigr).
\end{equation}

Basically, \citet{mammen1997locally} showed the asymptotic convergence of the solutions of the two problems when the distances among $x_{1:n}$ are close enough. In Theorem 9, they showed that, for then truth $g_{0,n}$ in a generic function class $\mathcal{G}$ with $\mathrm{TV}$-penalty, if one can choose a good enough linear subspace $\mathcal{G}_{n,k}$ with an oracle $g_{1,n}$, then the solution $\hat{g}$ on $\mathcal{G}_{n,k}$ preserves the optimal $L_2$ convergence on $\mathcal{G}$. Thus, the asymptotic conclusion holds by choosing $\mathcal{G}$ with $\mathrm{TV}$-penalty to be $\mathcal{F}_{k}$, and $\mathcal{G}_{n,k}$ to be $\mathcal{F}_{n,k}$.

\paragraph{The Equivalency of Assumption:}
Here, we analyze the relationship between LAS and univariate HAL from the perspective of assumption. We claim that, by the representation Theorem \ref{thm:HAL*representation}, $D^{(k)}_M\bigl([0,1]\bigr)$ is the closure of the infinite linear combination of k-order truncated power splines, denoted as $\overline{\mathcal{F}*{\infty,k}}$. Thus, we conclude that the solution on the finite-dimensional working model $\mathcal{F}_{n,k}$ is a finite-dimensional approximation of the projection of the truth $f_0$ on $D^{(k)}_M\bigl([0,1]\bigr)$, corresponding to the approximation of the integral in Section \ref{Estimation Construction}.

\textbf{For $k = 0$:} $f \in \mathcal{F}_{k}$ is not necessarily \emph{c\`adl\`ag}. For example, a function with finite jumps at particular points is still bounded total variation but not \emph{c\`adl\`ag}. However, after proving Theorem 9, \citet{mammen1997locally} claimed that, by Proposition 4, one can assume WLOG that the truth lies in the $\mathcal{F}_{n,k}$, and thus in $\overline{\mathcal{F}_{\infty,k}}$. However, this claim only holds since their results primarily concern $L_2$ convergence, whereas we are concerned with the \emph{c\`adl\`ag} condition for more pointwise behaviors, as elaborated in Section~\ref{Theoretical Properties}. The $0$-th sectional variational norm is the existence of $|f(0)|$ and the total variation bound.

\textbf{For $k \ge 1$:} The \emph{c\`adl\`ag} assumption and the existence of $|f^{(i)}(0)|$ for all $1 \le i \le k$ is implied by differentiability. So the assumption is equivalent.

\textbf{\textit{Remark:}} We could argue that we only need the Lebesgue–Radon–Nikodym derivative, and relax our assumption by a null set with Lebesgue measure 0. But they could be augmented in a similar manner as well. This is the reason \citet{tibshirani2014adaptive} introduced weak differentiability for the function class and the working model in Section 3.

\paragraph{The Focus on \{0\}-Section:}
The idea of \{0\}-Section arises from the degeneration of HAL to the univariate cases, where the definition of the $k$-th sectional variational norm is simplified as in (iii) of Assumption \ref{assump:DkM}. Apart from the traditional variational norm, the $k$-th sectional variational norm still requires the derivatives at $0$ to be well-defined for all orders.

The intuition is that not only do we want the function to preserve nice properties on $(0, 1]$, we also want it to be nice at point $0$. Here is a concrete example:
\[
f(x)=
\begin{cases}
\sin\bigl(1/x\bigr), & x\in(0,1],\\[6pt]
0, & x=0.
\end{cases}
\]
This function is also used in the simulation of \cite{mammen1997locally}. It is continuous and thus \emph{c\`adl\`ag}, but not bounded total variation. Its $k$-th order derivative at $0$ is also not well-defined.

\paragraph{The Generalized Lasso Form Optimizer:}

In Formula \ref{equation: HAL beta optimizer}, we formulated HAL as a generalized lasso problem.
\citet{tibshirani2014adaptive} also formulated the restricted local adaptive spline and trend filtering as generalized lasso problems. Starting from the truncated power basis, the total variation of the $k$-th order derivative of the estimation of the true function becomes a linear combination of the total variation of the $k$-th order derivative of the basis. Thus, the penalization of the function becomes the penalization of the coefficients of the `local" or `nonparametric'' part of the spline basis. In the same paper, \citet{tibshirani2014adaptive} then proposed trend filtering as a theoretically equivalent but computationally economical method. Trend filtering introduces the ideas of generalized difference and discrete splines back to this variational optimization, and replaces the truncated power basis with the falling factorial basis. Chapter 10 of \cite{tibshirani2022divided} shows that for each spline spanned by the truncated power basis, there is a close-enough spline, in terms of supremum norm, spanned by the falling factorial basis with the same total variation.

\paragraph{Similarity and Difference in Implementations:}
\label{Similarity and Difference in Implementations:}

In the univariate setting, \citet{tibshirani2014adaptive} showed that for spline orders \(k=0\) and \(k=1\), trend filtering and restricted locally adaptive regression splines coincide; they diverge for higher orders. Moreover, univariate HAL and restricted locally adaptive splines utilize the same truncated power basis of order \(k\), differing only by a constant scaling of the basis coefficients.

The principal implementation distinction lies in the penalty: the restricted locally adaptive spline’s generalized lasso formulation excludes the global, or nonparametric, part of the polynomial components from penalization \citep{tibshirani2014adaptive}, whereas HAL imposes an \(L_1\)‑penalty on all the basis coefficients. The difference in penalty is caused by the difference in definitions of sectional variational norm and total variation. We impose an extra penalty on the $\{0\}$-section. The difference will be significantly enlarged when it comes to the higher-dimension case.

\paragraph{Conclusion and Extension:}
HAL, restricted local adaptive spline, and trend filtering ended up solving similar Lasso problems, even though they initiated very differently. All these methods, along with wavelet \cite{donoho1998minimax}, achieve the optimal $L_2$ minimax error rate for the whole TV class in the univariate case. From a theoretical standpoint, HAL also achieves uniform convergence \citep{van2017uniform}.

Furthermore, for the extension to the density estimation problem, HAL Maximum Likelihood Estimator (HAL‑MLE) exhibits asymptotic normality, and thus, in Section~\ref{Variance Estimation} we propose a variance estimator for the HAL‑MLE. \citet{sadhanala2024exponential} proposed exponential family trend filtering for lattice data, assuming each observation was drawn from a different distribution within the natural exponential family. However, to the best of our knowledge, there is no density estimation technique related to restricted LAS and trend filtering for the aforementioned survival setup and censored data.

\subsection{HAL Density Estimation} \label{HAL Density Estimation}
In this section, we will extend the HAL regression to HAL-MLE by changing the loss function $L$ from MSE to log-likelihood. We will define a family of HAl-MLE and provide theorems for the $L_2$ convergence, uniform convergence, and pointwise asymptotic normality.

We start by interpreting Theorem \ref{thm:HAL_representation}. It shows that we can represent any function $f \in D^{(k)}_M\bigl([0,1]\bigr)$ by an infinite linear combination of the $k$-th order splines (bases). For the nonparametric part of the basis system, we index them each by a tuple $(k, u)$ for $k$-th order splines and any knot point $u \in (0,1]$. For the parametric part of the basis system, we index them each by a tuple $(i, 0)$, for $i \leq k$ and starting point 0. As such, we could define the total index set as a union of all these.

\begin{definition}[Index Set for Basis]\label{Index Set}
Let $\mathcal{R}_1^k = \{(k, u): k \text{ is fixed}, u \in (0, 1]$ \}, and let $\mathcal{R}_2^k = \{(i, 0): i \text{ is an integer}, i \leq k \}$. We define the total index set, which is an infinite set, as $\mathcal{R}^k = \mathcal{R}_1^k \cup \mathcal{R}_2^k$. We then define $D^{(k)}_M(\mathcal{R}^k)$ as the closure of the linear combination of splines indexed by the $\mathcal{R}^k$.
\end{definition}
\textbf{\textit{Remark:}} Based on Theorem \ref{thm:HAL_representation}, $D^{(k)}_M(\mathcal{R}^k) = D^{(k)}_M\bigl([0,1]\bigr)$. \\

Now, we define the finite-dimensional working model with J-knot points.
\begin{definition}[Index Set for Working Model]\label{Index Set}
We define the working model index set, which is a finite subset of $\mathcal{R}^k$, as $\mathcal{R}^k(J)$. Based on the knot points set $\{u_1, \cdots, u_J\}$, $\mathcal{R}_1^k(J) = \{(k, u): k \text{ is fixed}, u \in \{u_1, \cdots, u_J\} \}$.
\[
\mathcal{R}^k(J) = \mathcal{R}\_1^k(J) \cup \mathcal{R}\_2^k.
\]
We then define $D^{(k)}_M(\mathcal{R}^k(J))$ as the linear combination of splines indexed by the $\mathcal{R}^k(J)$.
\end{definition}

\textbf{\textit{Remark:}} When using a sample $x_{1:n}$ as the knot points set, we denote the working model index set as $\mathcal{R}^k(n)$. This is referred to as the initial working model, whereas the model after Lasso selection is called the post-selection working model, denoted as $\mathcal{R}^k(J_n)$.

Now, we define a family of HAL-MLE.

\begin{definition}[HAL-MLE] \label{Def:HAL-MLE}
The \emph{Highly Adaptive Lasso Maximum Likelihood Estimator (HAL-MLE)} for $f_0$ in $D^{(k)}_M\bigl([0,1]\bigr)$, with a user-supplied constant $M$ and log-likelihood function $L$, denoted as \( f*{n,\beta_n(M)} \), is defined as  
\begin{equation}
f*{n,\beta*n(M)} = \arg\max*{f*{n,\beta} \in D^{(k)}\_M\bigl(\mathcal{R}^k(n)\bigr)} P_n L(f*{n,\beta}),
\end{equation}
where $D^{(k)}_M\bigl(\mathcal{R}^k(n)\bigr)$ is the finite linear combination of the bases indexed by $\mathcal{R}^k(n)$ with the upper bound of the \( k \)-th order sectional variation norm $M$.
\end{definition}

\textbf{\textit{Remark 1:}} Notice that for a difference choice of $M$, we ended up with different post-selection models. However, they are all subsets of the same initial working model and thus indexed by it. Choosing among different initial working models is called Sieved HAL-MLE, which is not related to this paper.

\textbf{\textit{Remark 2:}} The choice of the initial working model makes a difference in the performance. The theoretical best choice of the initial working model is called the oracle, denoted as $f_{n,\beta_0(M)}$. Note that the oracle in the working model $f_{n,\beta_0(M)}$ is not the truth $f_0$. However, we will show that $f_{n,\beta_n(M)}$ is close enough to $f_{n,\beta_0(M)}$, and $f_{n,\beta_0(M)}$ is close to $f_0$ with a bit undersmoothening. We will define CV-HAL-MLE first and then define undersmoothened HAL-MLE.

\begin{definition}[CV-HAL-MLE] \label{Def:CV-HAL-MLE}
We define the CV-HAL-MLE to be
\begin{equation}
f*{n,\beta_n(M*{cv})} = \arg\max*{f*{n,\beta} \in D^{(k)}_{M_{cv}}\bigl(\mathcal{R}^k(n)\bigr)} P*n L(f*{n,\beta}),
\end{equation}
where $M_{cv}$ is chosen by the cross-validation selector from a candidate set of $M$.
\end{definition}

\textbf{\textit{Remark:}} In this paper, we will skip the introduction to cross-validation. One can propose the candidate set of $M$ based on the histogram or some other naive locally smooth density estimation.

\begin{definition}[Undersmoothened HAL-MLE] \label{Def:Undersmoothened HAL-MLE}
We define the undersmoothened HAL-MLE to be
\begin{equation}
f*{n,\beta_n(M')} = \arg\max*{f*{n,\beta} \in D^{(k)}*{M'}\bigl(\mathcal{R}^k(n)\bigr)} P*n L(f*{n,\beta}),
\end{equation}
for some $M' > M_{cv}$.
\end{definition}

\textbf{\textit{Remark:}} The post-selection working model changes since $M' > M_{cv}$, but still indexed by the initial working model. The undersmoothened HAL-MLE reduces bias by including more bases, but increases variance based on the trade-off.

And, finally, we introduce Relaxed HAL-MLE.

\begin{definition}[Relaxed HAL-MLE] \label{Def:Relaxed HAL-MLE}
We define the Relaxed HAL-MLE to be
\begin{equation}
f^{\text{relax}}_{n,\beta_n(M_{cv})} = \arg\max*{f*{n,\beta} \in D^{(k)}\bigl(\mathcal{R}^k(n)\bigr)} P*n L(f*{n,\beta}),
\end{equation}
where $D^{(k)}\bigl(\mathcal{R}^k(n)\bigr)$ is the finite linear combination of the bases indexed by $\mathcal{R}^k(n)$.
\end{definition}
This is achieved by refitting the working model selected by CV-HAL-MLE.

\subsection{Theoretical Properties for HAL-MLE} \label{Theoretical Properties}
In this section, we will demonstrate the $L_2$ convergence, pointwise asymptotic normality, and uniform convergence for the HAL-MLE family and its plug-ins.

Slightly abusing the notation, we will denote HAL-MLE, $f_{n, \beta_n}$, as $f_n$. And it could be represented as follows:
\begin{equation}
f*n = \sum*{j \in \mathcal{R}^k(J_n)} \beta_n(j) \phi_j,
\end{equation}
where $\beta_n(j)$ is the $j$-th entry of $\beta_n$ for $f_n$.

\begin{theorem}[Pointwise Asymptotic Normality for HAL-MLE]
\label{thm:density*HAL_asymptotic_normality}
Consider the \( k \)-th order HAL-MLE, CV-HAl-MLE, or relax HAL-MLE, \( f_n \), for the truth \(f_0 \in D^{(k)}\_M\bigl([0,1]\bigr)\). Let $d_n$ be the effective dimension of $D^{(k)}\bigl(\mathcal{R}^k(J_n)\bigr)$, that is the number of orthonormal basis $\phi^*\_j$ needed to span $D^{(k)}\bigl(\mathcal{R}^k(J_n)\bigr)$. Let $\tilde{\sigma}*{0,n} = \frac{1}{d*n} \sum*{j \in \mathcal{R}^k(J*n)} (\phi^\*\_j)^2$. Under the mild conditions specified in Appendix~\ref{Appendix A.2: Pointwise Asy for HAL-MLE},
\begin{equation}
\tilde{\sigma}\*{0,n}^{-1} \left( \frac{n}{d\_{0,n}} \right)^{1/2} \bigl(f_n - f_0\bigr)(x) \Rightarrow_d N(0,1).
\end{equation}
\end{theorem}

\begin{theorem}[Uniform Convergence for density]
\label{thm:uniform*convergence_density}
Under the same setting in Theorem~\ref{thm:density_HAL_asymptotic_normality} and some slightly stronger conditions in Appendix~\ref{Appendix A.3: Uniform conv for HAL-MLE}. Then the following uniform convergence results hold:
\begin{equation}
\sup*{x \in [0,1]} \left( \frac{n}{d*{n}} \right)^{1/2} \frac{|f_n - f_0|(x)}{\tilde{\sigma}*{0,n}(x)} = o*P(\log n).
\end{equation}
Furthermore, $\exists m < \infty$, such that
\begin{equation}
\|f_n - f_0\|*{\infty} = O_P\Bigl(n^{-(k+1)/(2k+3)} \log^m n\Bigr).
\end{equation}
\end{theorem}

Marginal survival function is a typical example of a pathwise differentiable functional. The definition of pathwise differentiability is included in Theorem A.1 in Appendix A.4 of \citet{van2011targeted}. We will use the Functional Delta Method to demonstrate the asymptoticity and uniform convergence of plug-in HAL-MLE with a pathwise differentiable functional. A formal proof is provided in Appendix~\ref{Appendix A.4: Plug-in HAl-MLE}.

\begin{definition}[Plug‐in Functional]\label{def:plugin*functional}
Let \(\Phi\) be a functional defined on a space of candidate densities. Given estimators \(f_n\) and a reference \(f*{0,n}\) (usually chosen to be the oracle), the \emph{plug‐in estimate} of \(\Phi\) is $ \Phi(f*n)$ and $ \Phi\bigl(f*{0,n}\bigr),$ respectively.

Assume that \(\Phi\) is pathwise differentiable at \(f*{0,n}\) with derivative
\(\,d\Phi\bigl(f*{0,n}\bigr)\). Then its first‐order (linear) expansion around \(f*{0,n}\) is
\[
\Phi(f_n) - \Phi\bigl(f*{0,n}\bigr)
= d\Phi\bigl(f*{0,n}\bigr)\bigl(f_n - f*{0,n}\bigr) + R*{\Phi,n},
\]
where \(R*{\Phi,n}\) is a higher‐order remainder term.
\end{definition}

Based on the first-order expansion, we notice that if 1)
\(d\Phi\bigl(f*{0,n}\bigr)\) is a bounded linear functional in a neighborhood of \(f*{0,n}\), and 2) the higher order remainder \(R\_{\Phi,n}\) is well behaved, then the pointwise asymptotic normality of HAL-MLE could be extended to its plug-in.

\begin{theorem}[Asymptotic Normality of Plug-in HAL-MLE]
\label{thm:plugin_HAL_asymptotic_normality}
Consider the same setting in Theorem~\ref{thm:density_HAL_asymptotic_normality} and extra assumptions on plug-in functional $\Phi$, as detailed in Appendix~\ref{Appendix A.4: Plug-in HAl-MLE}. We have the following statement:

\[
\sqrt{\tfrac{n}{d*{n}}}
\bigl[\Phi(f_n) - \Phi\bigl(f*{0}\bigr)\bigr]
\;\;\xrightarrow{d}\;\; N\bigl(0,\,\sigma^2\bigr),
\]
where \(\sigma^2\) is the asymptotic variance determined by the covariance of \(d\Phi\bigl(f\_{0,n}\bigr)\) under the true data-generating process.
\end{theorem}

\begin{theorem}[Uniform Convergence for Plug-In HAL-MLE]
\label{thm:uniform*convergence_plugin}
Consider the same setting in Theorem~\ref{thm:plugin_HAL_asymptotic_normality} with slightly stronger assumptions, as detailed in Appendix~\ref{Appendix A.4: Plug-in HAl-MLE}. We have the following uniform convergence for the plug-in estimator:
\[
\sup*{x \in [0,1]}
\bigl|\Phi(f*n)(x) \;-\; \Phi(f_0)(x)\bigr|
\;=\;
o_P\!\Bigl(\bigl(d*{n}/n\bigr)^{1/2}\log n\Bigr).
\]

\end{theorem}

\subsection{Nonparametric Maximum Likelihood Estimation (NPMLE)}
Since the basis is determined by data, and we are not assuming a particular parametric form for the true density, HAL-MLE is considered to be a Nonparametric Maximum Likelihood Estimation (NPMLE) method.

The Kaplan–Meier (KM) estimator \citep{kaplan1958nonparametric} is the canonical nonparametric maximum‐likelihood estimator for the survival function under right censoring. Andersen et al.\ \citep{andersen2012statistical} included an NPMLE perspective of the KM estimator within the counting‐process framework. However, the KM estimator is confined to univariate, right‐censored data and directly targets the marginal survival function rather than the underlying density. As an NPMLE, it enjoys asymptotic normality, with its variance most commonly estimated via Greenwood’s formula \citep{greenwood1926natural}.

An important distinction between HAL‑MLE and the Kaplan–Meier (KM) estimator \citep{kaplan1958nonparametric} lies in the functional constraints each imposes. By the HAL representation theorem (Theorem~\ref{thm:HAL_representation}), HAL‑MLE produces estimates confined to the space of càdlàg functions with bounded sectional variation, ensuring the estimated density remains within this pre‑specified class. In contrast, the KM estimator, as an NPMLE for the survival function, imposes no smoothness or variation‑norm constraints: its survival estimate is a right‑continuous step function with jumps at observed failure times. Consequently, any density derived from the KM curve consists of discrete point masses at those knots, even when the true underlying density is continuous.

Another line of research examines NPMLE through the lens of statistical optimal transport, particularly for mixture models. Han et al.\ \citep{han2023nonparametric} analyze the convergence of the mixing‐distribution NPMLE under the Gaussian‐smoothed 1‐Wasserstein (GOT) distance, demonstrating that smoothing can markedly improve rates compared to classical unsmoothed metrics. While these optimal‐transport results provide valuable distributional convergence guarantees, our focus remains on the uniform convergence of HAL‑MLE within the bounded sectional‐variation‐norm class, and we therefore do not pursue the GOT perspective here.

\subsection{Targeted Maximum Likelihood Estimation (TMLE) for Survival}
Targeted maximum likelihood estimation (TMLE) is a general framework
for constructing semiparametric efficient substitution estimators for
pathwise‐differentiable parameters, with variants such as
cross‐validated TMLE (CV‐TMLE), collaborative TMLE (C‐TMLE). TMLE has been widely
applied in causal inference and survival analysis \citep{van2011targeted, van2006targeted}.
Given a pathwise‐differentiable functional \(\Psi(P)\), TMLE proceeds
by obtaining an initial estimate \(\hat P\) and then updating it along
a least‐favorable parametric submodel \(\{P*\epsilon\}\) chosen so that
the score at \(\epsilon=0\) equals the efficient influence function
\(D^\*(P)\). This fluctuation—implemented either as a one‐step update
or iteratively—solves the estimating equation
\[
\frac{1}{n}\sum*{i=1}^n D^_\bigl(O*i;\hat P*\epsilon\bigr) = 0,
\]
thereby achieving asymptotic efficiency. Common examples of
pathwise‐differentiable targets include the average treatment effect
and the marginal survival probability. In the uncensored univariate
case, for \(\Psi(P)=S(t_0)=P(T>t_0)\), the efficient influence function
is
\[
D^_(T) = I(T > t_0) - S(t_0).
\]

\subsection{EM algorithm}
The expectation–maximization (EM) algorithm \citep{dempster1977maximum} is a fundamental tool for handling censored data in survival analysis. Beginning with an initial density estimate—typically based on the uncensored failure times—the algorithm alternates between an \emph{E‑step}, which imputes the contribution of censored observations via their conditional expectations under the current fit, and an \emph{M‑step}, which updates the density by maximizing the expected complete‑data log‑likelihood as if the data were fully observed. These iterations continue until the observed‑data log‑likelihood converges. Under mild regularity conditions, the EM sequence converges to the unique nonparametric maximum‑likelihood estimator \citep{wu1983convergence,turnbull1976empirical}.

A simple analog for interval‑censored data in the univariate case
imputes each failure time to a representative point within its
censoring interval (e.g., the midpoint), assigns probability mass there, and then applies EM to reallocate mass until convergence
\citep{turnbull1976empirical}. In Section~\ref{Extension to Censored Data under Coarsening at Random}, we extend this approach to HAL‑MLE—denoted EM‑HAL‑MLE—to accommodate both right‑ and interval‑censored observations.

\section{Methodology for Density Estimation} \label{Density_section}
\subsection{Full Data}
\label{Full Data}
\subsubsection{HAL-MLE with Link Function}

Let the failure time T be defined on $[0,1]$ with distribution $P_T$. Suppose that we index the full-data density \( p*T = dP_T/d\mu \) in terms of a \( p*{T,f} \) for a function \( f \in D^{(k)}\_M([0,1])\), where \( \mu \) could be chosen as the Lebesgue measure for simplicity. Specifically, let

\[
\mathcal{M}^F = \{ p\_{T,f} : f \in \mathcal{F} \subset D^{(k)}\_M([0,1]) \}
\]

for some link function that maps \( f \) to \(p\_{T,f} \) and subset \( \mathcal{F} \subset D^{(k)}\_M([0,1]) \). Using the same link function in \citet{kooperberg1992logspline}:

\[
p*{T,f}(t) = \frac{\exp(f(t))}{\int*{t_1} \exp(f(t_1)) d\mu(t_1)},
\]

The normalizing constant in the denominator ensures that \( p*{T,f} \) integrates to 1. With the representation of HAL basis, we could denote $f$ to be $f*{n, \beta}$ for knot points determined by n sample points and coefficient $\beta$.

The intuition of this choice of link function is to assume that the log-likelihood of the density follows a càdlàg structure with bounded sectional variation, making it well-suited for modeling via HAL. This link function is also referred to as the \textbf{histo-spline method} \citep{leonard1978density, silverman1982estimation}. Specifically, when using \textbf{zero-order HAL basis functions}, the function \( f \) can be expressed as a histogram. Applying the transformation induced by the link function retains this histogram structure, making it a natural choice for density estimation.

One major benefit of this link function is that the resulting density belongs to the \textbf{exponential family}, ensuring that the log-likelihood is \textbf{strictly concave}. Consequently, when we apply \textbf{MLE with an \( L_1 \) norm constraint}, the optimization problem remains \textbf{convex}, leading to a \textbf{unique solution}.

If we observe \( T_1, \dots, T_n \sim P_T, i.i.d.\), then we define the full-data loss $L^F(f)$ to be the log-likelihood, and such that we have the following formula:

\[
P*{T,n} L^F(f) = \frac{1}{n} \sum*{i=1}^{n} \log p*f(T_i),
\] where $P*{T,n}$ denotes the empirical mean operator with respect to full distribution $T$.

The full-data HAL-MLE of the true \( f*{n, \beta} \) is then denoted as \(f*{n, \beta_n(M)}\) where

\[
\beta*n(M) = \arg \max*{\beta, \|\beta\|_1 < M} \sum_{i=1}^{n} L^F(f*{n, \beta})(T_i) = \arg \max*{\beta, \|\beta\|\_1 \leq M} \sum_j \beta_j \phi_j(T_i) - \log C(\beta),
\] where the $C(\beta)$ is the normalizing constant. If $M$ is chosen by cross-validation, then this whole procedure fits Definition \ref{Def:CV-HAL-MLE}. An algorithm demonstration is in the Appendix \ref{appendix:A3} Algorithm \ref{alg:1}.
