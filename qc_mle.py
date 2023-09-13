import numpy as np
from scipy.optimize import minimize

# Simulate some sample data for the manufacturing process (you can replace this with real data)
np.random.seed(0)
data = np.random.normal(loc=10, scale=2, size=100)

# Define the likelihood function
def likelihood(params, data):
    mu = params[0]  # Parameter to estimate (mean)
    n = len(data)
    log_likelihood = -n/2 * np.log(2 * np.pi) - n/2 * np.log(params[1]**2) - 1 / (2 * params[1]**2) * np.sum((data - mu)**2)
    return -log_likelihood  # We minimize the negative log-likelihood

# Initial guess for the parameter (mean) and standard deviation
initial_guess = [8, 2]

# Use scipy's minimize function to find the MLE estimate
result = minimize(likelihood, initial_guess, args=(data,), method='Nelder-Mead')

if result.success:
    estimated_mean, estimated_std = result.x
    print(f"MLE Estimate of Mean: {estimated_mean:.2f}")
    print(f"MLE Estimate of Standard Deviation: {estimated_std:.2f}")
else:
    print("MLE estimation failed.")

# Perform quality control checks based on the MLE estimates
if result.success:
    target_mean = 10  # Target mean for the manufacturing process
    tolerance = 0.5  # Tolerance for quality control
    
    if abs(estimated_mean - target_mean) <= tolerance:
        print("Quality Control: The process is within tolerance.")
    else:
        print("Quality Control: The process is out of tolerance. Investigate further.")
