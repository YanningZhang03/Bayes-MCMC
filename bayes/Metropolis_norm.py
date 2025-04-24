import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import time


# ======= 读取数据 =======
df = pd.read_csv('bayes_dataset.csv')  # 确保数据文件在当前目录
x = df['x'].values.astype(float)
y = df['y'].values.astype(int)
n = len(y)

# ======= 模型参数（用于可视化对比）=======
true_beta0 = 1.3862
true_beta1 = -0.1880

# ======= 先验参数 =======
mu_0, mu_1 = 0, 0
sigma_0, sigma_1 = 100, 100


# ======= 后验（未标准化） =======
def posterior(beta):
    beta0, beta1 = beta
    linear = beta0 + beta1 * x
    prob = norm.cdf(linear)
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    likelihood = np.prod(prob ** y * (1 - prob) ** (1 - y))
    prior = norm.pdf(beta0, mu_0, sigma_0) * norm.pdf(beta1, mu_1, sigma_1)
    return likelihood * prior


# ======= Fisher 信息矩阵估计（在初始点处）=======
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


# ======= 设置多元正态建议分布协方差矩阵 =======
beta_init = np.array([0.0, 0.0])
Fisher = fisher_info(beta_init)
proposal_cov = np.linalg.inv(Fisher + np.eye(2) * 1e-6)  # 防止奇异
proposal_cov *= 0.5  # 缩放因子调整接受率

# ======= Metropolis-Hastings 采样（多元正态建议分布）=======
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

end_time = time.time()  # 在采样结束后记录时间
print(f"采样耗时: {end_time - start_time:.2f} 秒")
# ======= 可视化结果 =======
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(samples[:, 0], bins=30, density=True, alpha=0.7, label=r'$\beta_0$')
plt.axvline(true_beta0, color='r', linestyle='--', label='True $/beta_0$')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(samples[:, 1], bins=30, density=True, alpha=0.7, label=r'$\beta_1$')
plt.axvline(true_beta1, color='r', linestyle='--', label='True $/beta_1$')
plt.legend()

plt.suptitle('Posterior Samples of β using Probit + MH (Fisher Proposal)')
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

