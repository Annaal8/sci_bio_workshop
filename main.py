#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import integrate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import sympy as sp

def print_library_versions():
    """Print versions of all required libraries to verify they are installed."""
    import numpy, scipy, sklearn, seaborn, matplotlib, pandas, sympy
    print("=== Library versions ===")
    print("numpy      :", numpy.__version__)
    print("scipy      :", scipy.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("seaborn    :", seaborn.__version__)
    print("matplotlib :", matplotlib.__version__)
    print("pandas     :", pandas.__version__)
    print("sympy      :", sympy.__version__)
    print("========================\n")


def main():
    print_library_versions()

    # --- SymPy: define a symbolic function ---
    x = sp.symbols('x')
    f_expr = sp.exp(-x**2) * sp.sin(3 * x)  # smooth "nice" function

    # Lambdify to numerical function for numpy and math (for scipy.integrate)
    f_np = sp.lambdify(x, f_expr, "numpy")
    f_math = sp.lambdify(x, f_expr, "math")

    # --- NumPy: generate data ---
    rng = np.random.default_rng(42)
    X = np.linspace(-3, 3, 200)
    y_true = f_np(X)
    y_noisy = y_true + rng.normal(scale=0.1, size=X.shape)

    # --- SciPy: compute area under the curve ---
    area, err = integrate.quad(f_math, -3, 3)

    # --- scikit-learn: polynomial regression to smooth noisy data ---
    poly = PolynomialFeatures(degree=10)
    X_poly = poly.fit_transform(X.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_poly, y_noisy)
    y_pred = model.predict(X_poly)

    # --- pandas: put everything into a DataFrame ---
    df = pd.DataFrame(
        {
            "x": X,
            "y_true": y_true,
            "y_noisy": y_noisy,
            "y_pred": y_pred,
        }
    )

    # --- seaborn + matplotlib: nice plot ---
    sns.set_theme(style="darkgrid")

    plt.figure(figsize=(10, 6))

    # Scatter noisy points
    sns.scatterplot(
        data=df,
        x="x",
        y="y_noisy",
        s=20,
        label="Noisy samples",
    )

    # True function and regression curve
    plt.plot(df["x"], df["y_true"], linewidth=2, label="True function (SymPy → NumPy)")
    plt.plot(
        df["x"],
        df["y_pred"],
        linestyle="--",
        linewidth=2,
        label="Polynomial regression (scikit-learn)",
    )

    plt.title(
        "Demo plot using numpy / scipy / scikit-learn / seaborn / matplotlib / pandas / sympy\n"
        f"Area under true curve on [-3, 3] ≈ {area:.3f} (estimated with SciPy)"
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()