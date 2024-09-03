# https://chatgpt.com/c/a766271e-37bc-4eec-8f06-fd03008db975

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 10000

# Distributions to consider
distributions = {
    'Normal': np.random.normal(0, 1, n_samples),
    'Uniform': np.random.uniform(-1, 1, n_samples),
    'Exponential': np.random.exponential(1, n_samples),
    'Laplace': np.random.laplace(0, 1, n_samples),
    'Cauchy': np.random.standard_cauchy(n_samples)
}

# Prepare the plots
fig, axs = plt.subplots(5, 2, figsize=(15, 20))

# Loop over the distributions and compute metrics
for i, (name, data) in enumerate(distributions.items()):
    # Compute STD and EVAD
    std_dev = np.std(data)
    evad = np.mean(np.abs(data - np.mean(data)))
    
    # Plot histogram
    sns.histplot(data, bins=50, kde=True, ax=axs[i, 0])
    axs[i, 0].set_title(f'{name} Distribution')
    axs[i, 0].axvline(np.mean(data), color='red', linestyle='dashed', linewidth=1)
    axs[i, 0].axvline(np.mean(data) + std_dev, color='blue', linestyle='dashed', linewidth=1)
    axs[i, 0].axvline(np.mean(data) - std_dev, color='blue', linestyle='dashed', linewidth=1)
    
    # Display STD and EVAD
    axs[i, 1].bar(['STD', 'EVAD'], [std_dev, evad], color=['blue', 'orange'])
    axs[i, 1].set_title(f'STD vs EVAD for {name} Distribution')
    axs[i, 1].set_ylim(0, max(std_dev, evad) * 1.5)

plt.tight_layout()
plt.show()
