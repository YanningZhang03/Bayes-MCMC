import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import arviz as az
import time

df = pd.read_csv('bayes_dataset.csv')
x = df['x'].values.astype(float)
y = df['y'].values.astype(int)
n = len(y)

MLE_beta0 = 1.3862
MLE_beta1 = -0.1880

mu_0, mu_1 = 0, 0
sigma_0, sigma_1 = 100, 100

def posterior(beta):
    beta0, beta1 = beta
    linear = beta0 + beta1 * x
    prob = norm.cdf(linear)
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    likelihood = np.prod(prob ** y * (1 - prob) ** (1 - y))
    prior = norm.pdf(beta0, mu_0, sigma_0) * norm.pdf(beta1, mu_1, sigma_1)
    return likelihood * prior

def fisher_info(beta):
    beta0, beta1 = beta
    eta = beta0 + beta1 * x
    p = norm.cdf(eta)
    phi = norm.pdf(eta)
    weight = (phi ** 2) / (p * (1 - p))

    I00 = np.sum(weight)
    I01 = np.sum(weight * x)
    I11 = np.sum(weight * x ** 2)

    return np.array([[I00, I01],
                     [I01, I11]])

beta_init = np.array([0.0, 0.0])
Fisher = fisher_info(beta_init)
proposal_cov = np.linalg.inv(Fisher + np.eye(2) * 1e-6)
proposal_cov *= 0.5

n_samples = 50000
samples = np.zeros((n_samples, 2))
beta_curr = beta_init.copy()
start_time = time.time()

for t in range(n_samples):
    beta_prop = np.random.multivariate_normal(mean=beta_curr, cov=proposal_cov)
    p_curr = posterior(beta_curr)
    p_prop = posterior(beta_prop)

    alpha = min(1, p_prop / p_curr)
    if np.random.rand() < alpha:
        beta_curr = beta_prop
    samples[t] = beta_curr

end_time = time.time()
print(f"采样耗时: {end_time - start_time:.2f} 秒")


# 计算后验期望和HPD区间
idata = az.convert_to_inference_data(samples, coords={"param": ["beta0", "beta1"]}, dims={"chain": ["param"]})
summary = az.summary(idata, hdi_prob=0.9)

# 可视化：直方图 + 真值
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

# 样本路径图
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

# 遍历均值图
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

# 散点图
plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=1)
plt.xlabel(r'$\beta_0$')
plt.ylabel(r'$\beta_1$')
plt.title('Joint Posterior Samples')
plt.grid(True)
plt.tight_layout()
plt.show()

# Gelman-Rubin（多链R-hat）诊断（模拟多链）
multi_chains = np.reshape(samples[:40000], (4, 10000, 2))
idata_multi = az.from_dict(posterior={"beta": multi_chains})
gr_stats = az.rhat(idata_multi)

# Monte Carlo 标准误差估计
mcse = az.mcse(idata)

summary, gr_stats, mcse  # 返回摘要统计、GR诊断和MC误差


