# t-test
# McNemar test

import numpy as np
from scipy.stats import ttest_rel

# Sample data for paired t-test
before = np.array([4, 2, 7, 6, 3, 5, 8, 6, 9, 2])
after = np.array([5, 6, 8, 7, 4, 6, 9, 7, 2, 3])

# Calculate the paired t-test
t_statistic, p_value = ttest_rel(before, after)

# Print the results
print("Paired t-test results:")
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")