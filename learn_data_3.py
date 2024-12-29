import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
from pylab import *
 
mu, sigma = 5, 0.7
lower, upper = mu - 2 * sigma, mu + 2 * sigma  # 截断在[μ-2σ, μ+2σ]
X = stats.truncnorm(-3, 3, loc=0.25, scale=0.25/3)
s = X.rvs(10000)
print(s.mean())
plt.hist(s, bins=30, density=True, alpha=0.6, color='g')