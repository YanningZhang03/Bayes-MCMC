import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import time
import arviz as az

# 读取数据
df = pd.read_csv('bayes_dataset.csv')
x = df['x'].values.astype(float)
y = df['y'].values.astype(int)
n = len(y)
MLE_beta0 = 1.3862
MLE_beta1 = -0.1880

# 先验参数
mu_0, mu_1 = 0, 0
sigma_0, sigma_1 = 100, 100
tau_0, tau_1 = 0.1, 0.1

# 后验概率函数（非标准化）
def posterior(beta):
    beta0, beta1 = beta
    linear = beta0 + beta1 * x
    prob = norm.cdf(linear)
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    likelihood = np.prod(prob**y * (1 - prob)**(1 - y))
    prior = norm.pdf(beta0, mu_0, sigma_0) * norm.pdf(beta1, mu_1, sigma_1)
    return likelihood * prior

# Metropolis 采样
n_samples = 50000
samples = np.zeros((n_samples, 2))
beta_curr = np.array([0.0, 0.0])
start_time = time.time()

for t in range(n_samples):
    beta_prop = beta_curr + np.random.normal(0, [tau_0, tau_1])
    p_curr = posterior(beta_curr)
    p_prop = posterior(beta_prop)
    alpha = min(1, p_prop / p_curr)
    if np.random.rand() < alpha:
        beta_curr = beta_prop
    samples[t] = beta_curr

end_time = time.time()
print(f"Sampling time: {end_time-start_time:.2f} s")

idata = az.from_dict(posterior={"beta": samples[np.newaxis, :, :]})
summary = az.summary(idata, hdi_prob=0.9)
print(summary)

multi_chains = samples[:16000].reshape(4, 4000, 2)
idata_multi = az.from_dict(posterior={"beta": multi_chains})
gr_stats = az.rhat(idata_multi)
print(gr_stats)

mcse = az.mcse(idata)
print(mcse)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(samples[:, 0], bins=30, density=True, alpha=0.7, label=r'$\beta_0$')
plt.axvline(MLE_beta0, color='r', linestyle='--', label=r'MLE $\beta_0$')
plt.axvline(np.mean(samples[:, 0]), color='g', linestyle='--', label='Posterior Mean')
plt.legend()
plt.subplot(1, 2, 2)
plt.hist(samples[:, 1], bins=30, density=True, alpha=0.7, label=r'$\beta_1$')
plt.axvline(MLE_beta1, color='r', linestyle='--', label=r'MLE $\beta_1$')
plt.axvline(np.mean(samples[:, 1]), color='g', linestyle='--', label='Posterior Mean')
plt.legend()
plt.suptitle('Posterior Histograms with True Values and Means')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(samples[:, 0], lw=0.5)
plt.title(r'Trace of $\beta_0$')
plt.xlabel('Iteration')
plt.ylabel(r'$\beta_0$')
plt.subplot(1, 2, 2)
plt.plot(samples[:, 1], lw=0.5)
plt.title(r'Trace of $\beta_1$')
plt.xlabel('Iteration')
plt.ylabel(r'$\beta_1$')
plt.suptitle('Trace Plots of MCMC Samples')
plt.tight_layout()
plt.show()

cum_mean = np.cumsum(samples, axis=0) / np.arange(1, n_samples + 1).reshape(-1, 1)
plt.figure(figsize=(12, 5))
plt.plot(cum_mean[:, 0], label=r'Cumulative Mean $\beta_0$')
plt.plot(cum_mean[:, 1], label=r'Cumulative Mean $\beta_1$')
plt.axhline(MLE_beta0, color='r', linestyle='--', label=r'MLE $\beta_0$')
plt.axhline(MLE_beta1, color='b', linestyle='--', label=r'MLE $\beta_1$')
plt.legend()
plt.title("Cumulative Means of Posterior Samples")
plt.xlabel("Iteration")
plt.ylabel("Mean")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=1)
plt.xlabel(r'$\beta_0$')
plt.ylabel(r'$\beta_1$')
plt.title('Joint Posterior Samples')
plt.grid(True)
plt.tight_layout()
plt.show()
