import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Linear relationship data
x_linear = np.linspace(0, 10, 100)
y_linear = 2 * x_linear + np.random.normal(0, 1, 100)

# Non-linear relationship data (quadratic)
x_nonlinear = np.linspace(0, 10, 100)
y_nonlinear = x_nonlinear**2 + np.random.normal(0, 10, 100)

# Calculate Pearson correlation
corr_linear, _ = pearsonr(x_linear, y_linear)
corr_nonlinear, _ = pearsonr(x_nonlinear, y_nonlinear)

# Plot the relationships
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_linear, y_linear)
plt.title(f'Linear Relationship\nPearson correlation: {corr_linear:.2f}')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 2, 2)
plt.scatter(x_nonlinear, y_nonlinear)
plt.title(f'Non-Linear Relationship\nPearson correlation: {corr_nonlinear:.2f}')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
