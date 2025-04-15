import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.optimize import LinearConstraint, minimize

from warnings import filterwarnings

filterwarnings("ignore", "divide by zero", category=RuntimeWarning)
filterwarnings("ignore", "invalid value", category=RuntimeWarning)
filterwarnings("ignore", "overflow encountered", category=RuntimeWarning)

################################################################### Callback Function ###################################################################
def callback(feature_names):
    iter = 0
    
    def display_param(xk, res=None):
        nonlocal iter # iter can be used in outer function but not globally
        iter += 1
        print(f"Iteration: {iter}")
        print("Current parameter values:")
        for name, value in zip(feature_names, xk):
            print(f"{name:<30} {value:.6f}")
        print("\n") 

    return display_param

################################################################### Linear Regression (OLS) ###################################################################
def sm_ols(X, y, method="qr"):
    ols = sm.OLS(y,X)
    model = ols.fit(method=method,
                    disp=True,
                    maxiter=1000)

    # Display final model parameters
    print("\nModel final parameters:")
    print(model.params)
    print("\nModel fitting summary:")
    print(model.summary())

    # Save the summary to a text file
    summary_path = f"ols_model_summary.txt"
    with open(summary_path, "w") as file:
        file.write(model.summary().as_text())
    print(f"Model summary saved to {summary_path}\n")

    # Optionally remove unnecessary data to reduce memory usage
    # model.remove_data()

    return model



def constrained_sm_ols(X, y, ols_result, thresh, method="trust-constr"):
    n_features = X.shape[1]
    sig = ols_result.pvalues
    sig.reset_index(drop=True, inplace=True)

    constraints_index = np.where(sig > thresh)[0]

    ncons = len(constraints_index)
    A = np.zeros((ncons, n_features))
    A[np.arange(ncons), constraints_index] = 1

    lb = ub = np.zeros(ncons)
    constraints = LinearConstraint(A, lb, ub)


    if method == "SLSQP":
        start_params = ols_result.params.to_numpy(copy=True).ravel(order="F")
    else:
        start_params = ols_result.params.to_numpy(copy=True)
    start_params[constraints_index] = 0

    callback_func = callback(X.columns)

    def objective(params):
        residuals = y - np.dot(X, params)
        return np.sum(residuals**2)
    
    result = minimize(objective, start_params, method=method, constraints=constraints, 
                      callback=callback_func, tol=5e-9, options={"maxiter": 1000, "disp":True})
    
    param_names = X.columns
    optimized_params = pd.Series(result.x, index=param_names)
    
    return optimized_params


################################################################### Linear Regression (GLM Gaussian) ###################################################################

def sm_glm_gaussian(X, y, method="IRLS"):
    glm = sm.GLM(y, X, family=sm.families.Gaussian())
    model = glm.fit(
        method=method,
        maxiter=1000,
        tol=1e-9
    )

    # Display final model parameters
    print("\nModel final parameters:")
    print(model.params)
    print("\nModel fitting summary:")
    print(model.summary())

    # Save the summary to a text file
    summary_path = f"GLM_Gaussian_summary.txt"
    with open(summary_path, "w") as file:
        file.write(model.summary().as_text())
    print(f"Model summary saved to {summary_path}\n")

    # Optionally remove unnecessary data to reduce memory usage
    # model.remove_data()

    return model

def constrained_sm_glm_gaussian(X, y, glm_gau_result, thresh):
    n_features = X.shape[1]
    sig = glm_gau_result.pvalues
    sig.reset_index(drop=True, inplace=True)


    constraints_index = np.where(sig > thresh)[0]
    ncons = len(constraints_index)

    R = np.zeros((ncons, n_features))
    R[np.arange(ncons), constraints_index] = 1
    q = np.zeros(ncons)

    start_params = glm_gau_result.params.to_numpy(copy=True)
    start_params[constraints_index] = 0

    constrained_model_glm = sm.GLM(y, X, family=sm.families.Gaussian())
    model = constrained_model_glm.fit_constrained(start_params=start_params,
                                                  constraints=(R,q))
    # Display final model parameters
    print("\nModel final parameters:")
    print(model.params)
    print("\nModel fitting p-values:")
    print(model.pvalues)
    print("\nModel fitting summary:")
    print(model.summary())

    summary_path = f"GLM_Gaussian_summary_constrained.txt"
    with open(summary_path, "w") as file:
        file.write(model.summary().as_text())
    print(f"Model summary saved to {summary_path}\n")

    # Optionally remove unnecessary data to reduce memory usage
    # model.remove_data()

    return model
