import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import time
# 读取 bayes_dataset.csv 数据
df = pd.read_csv('bayes_dataset.csv')  # 确保文件在当前工作目录下，或提供完整路径
x = df['x'].values.astype(float)  # 转为 float 以保证后续计算正常
y = df['y'].values.astype(int)
n = len(y)
true_beta0 = 1.3862
true_beta1 = -0.1880
p_true = norm.cdf(true_beta0 + true_beta1 * x)


# 先验参数
mu_0, mu_1 = 0, 0
sigma_0, sigma_1 = 100, 100
tau_0, tau_1 = 0.1, 0.1

# 目标函数：后验概率（非标准化）
def posterior(beta):
    beta0, beta1 = beta
    linear = beta0 + beta1 * x
    prob = norm.cdf(linear)
    prob = np.clip(prob, 1e-10, 1 - 1e-10)  # 防止数值下溢
    likelihood = np.prod(prob**y * (1 - prob)**(1 - y))
    prior = norm.pdf(beta0, mu_0, sigma_0) * norm.pdf(beta1, mu_1, sigma_1)
    return likelihood * prior


# 逐分量 Metropolis-Hastings 采样
n_samples = 50000
samples = np.zeros((n_samples, 2))
beta_curr = np.array([0.0, 0.0])
start_time = time.time()

for t in range(n_samples):
    for i in range(2):  # 逐分量更新 beta_0 和 beta_1
        beta_prop = beta_curr.copy()
        beta_prop[i] += np.random.normal(0, [tau_0, tau_1][i])

        p_curr = posterior(beta_curr)
        p_prop = posterior(beta_prop)
        alpha = min(1, p_prop / p_curr)

        if np.random.rand() < alpha:
            beta_curr[i] = beta_prop[i]

    samples[t] = beta_curr
end_time = time.time()  # 在采样结束后记录时间
print(f"采样耗时: {end_time - start_time:.2f} 秒")

# 可视化结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(samples[:, 0], bins=30, density=True, alpha=0.7, label=r'$\beta_0$')
plt.axvline(true_beta0, color='r', linestyle='--', label='True $/beta_0$')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(samples[:, 1], bins=30, density=True, alpha=0.7, label=r'$\beta_1$')
plt.axvline(true_beta1, color='r', linestyle='--', label='True $/beta_1$')
plt.legend()

plt.suptitle('Posterior Samples of β using Probit + Metropolis')
plt.tight_layout()
plt.show()
# ======= 样本路径图（Trace Plots） =======
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

