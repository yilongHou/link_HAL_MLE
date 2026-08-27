├── methods
│   ├── non_HAL_method
│   │   ├── KDE
│   │   │   └── estimator.py
│   │   ├── TF_ADMM
│   │   │   └──  estimator.py
│   ├── base_estimator.py # Base class for HAL estimators
│   ├── deep_learning_method
│   │   └── auto_diff
│   │       └── estimator.py
│   ├── first_order_method
│   │   ├── cvxpy
│   │   │   └── estimator.py
│   │   ├── fista
│   │   │   └── estimator.py
│   │   ├── projected_gradient_descent
│   │   │   └── estimator.py
│   │   └── proximal_gradient_descent
│   │       └── estimator.py
│   └── second_order_method
│       ├── proximal_adagrad
│       │   └── estimator.py
│       ├── proximal_newton
│       │   └── estimator.py
│       ├── proximal_newton_lbfgs
│       │   └── estimator.py
│       ├── proximal_newton_lbfgs_full
│       │   └── estimator.py
│       └── proximal_newton_sdcd
│           └── estimator.py