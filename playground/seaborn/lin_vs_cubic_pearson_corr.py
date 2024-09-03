import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

# Modify the relationship to introduce quadratic and cubic terms with a smaller linear term
x_mixed = np.linspace(0, 10, 100)
y_mixed = 0.1 * x_mixed + 0.1 * x_mixed**2 - 2 * x_mixed**3 + np.random.normal(0, 1, 100)

# Calculate Pearson correlation for the new relationship
corr_mixed, _ = pearsonr(x_mixed, y_mixed)

# Plot the relationship using seaborn
plt.figure(figsize=(6, 5))
sns.scatterplot(x=x_mixed, y=y_mixed)
plt.title(f'Complex Relationship (Linear, Quadratic, Cubic)\nPearson correlation: {corr_mixed:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.tight_layout()
plt.show()
