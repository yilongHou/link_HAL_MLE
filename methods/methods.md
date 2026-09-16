├── methods
│   ├── non_HAL_method
│   │   ├── KDE
│   │   │   └── estimator.py
│   │   ├── LOG_SPLINES              # wraps R `logspline` via rpy2 (optional extra)
│   │   │   └── estimator.py
│   │   ├── TF_ADMM                  # specialized ADMM of Appendix H; the reported TF results use TF_CVXPY
│   │   │   └── estimator.py
│   │   ├── TF_CVXPY                 # trend filtering; exported as TrendFilteringADMMEstimator
│   │   │   └── estimator.py
│   │   └── TF_CVXPY_PP              # trend filtering with post-processing (TFPP)
│   │       └── estimator.py
│   ├── base_estimator.py            # Base class for HAL estimators
│   ├── first_order_method
│   │   ├── cvxpy                    # HAL-MLE via CVXPY (MOSEK, with open-source fallbacks)
│   │   │   └── estimator.py
│   │   └── fista
│   │       └── estimator.py
│   └── second_order_method
│       ├── proximal_adagrad
│       │   └── estimator.py
│       ├── proximal_newton
│       │   └── estimator.py
│       └── proximal_newton_lbfgs
│           └── estimator.py
