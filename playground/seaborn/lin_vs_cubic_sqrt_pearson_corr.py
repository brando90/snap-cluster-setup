import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Generate data for a quadratic relationship
x = np.linspace(-2, 2, 100)

# Add an offset to ensure Y values are non-negative
offset = 0
y_quadratic_offset = x**2 + offset + np.random.normal(0, 5, 100)

# Transform Y to sqrt(Y)
sqrt_y_offset = np.sqrt(y_quadratic_offset)

# Calculate Pearson correlation
corr_transformed_offset, _ = pearsonr(x, sqrt_y_offset)

# Plot the relationship between X and sqrt(Y)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=x, y=sqrt_y_offset)
plt.title(f'Linear Relationship between X and sqrt(Y)\nPearson correlation: {corr_transformed_offset:.2f}')
plt.xlabel('X')
plt.ylabel('sqrt(Y)')
plt.tight_layout()
plt.show()
