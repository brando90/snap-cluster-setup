import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Set random seed for reproducibility
np.random.seed(42)

# Generate a dataset with a linear relationship
x = np.linspace(0, 10, 100)
y = 2 * x + np.random.normal(0, 1, 100)  # Linear with noise

# Introduce outliers in the data
x_outliers = np.append(x, [20, 22, 24])
y_outliers = np.append(y, [50, 55, 60])

# Standardize the data
x_standardized = (x_outliers - np.mean(x_outliers)) / np.std(x_outliers)
y_standardized = (y_outliers - np.mean(y_outliers)) / np.std(y_outliers)

# Calculate Pearson correlation with and without standardization
corr_with_outliers, _ = pearsonr(x_outliers, y_outliers)
corr_standardized, _ = pearsonr(x_standardized, y_standardized)

# Plot the relationship with outliers
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=x_outliers, y=y_outliers)
plt.title(f'With Outliers\nPearson correlation: {corr_with_outliers:.2f}')
plt.xlabel('X')
plt.ylabel('Y')

# Plot the standardized relationship with outliers
plt.subplot(1, 2, 2)
sns.scatterplot(x=x_standardized, y=y_standardized)
plt.title(f'Standardized with Outliers\nPearson correlation: {corr_standardized:.2f}')
plt.xlabel('Standardized X')
plt.ylabel('Standardized Y')

plt.tight_layout()
plt.show()
