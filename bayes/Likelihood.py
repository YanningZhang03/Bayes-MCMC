import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# 读取数据
df = pd.read_csv('bayes_dataset.csv')
x = df['x'].values.astype(float)
y = df['y'].values.astype(int)
n = len(y)

# 增加一列全1，用于beta0（截距项）
X = np.column_stack((np.ones(n), x))

# 定义负对数似然函数
def neg_log_likelihood(beta):
    linear = X @ beta  # 矩阵乘法：beta0 + beta1 * x
    prob = norm.cdf(linear)  # Probit模型：CDF of standard normal
    prob = np.clip(prob, 1e-10, 1 - 1e-10)  # 避免数值问题
    log_likelihood = y * np.log(prob) + (1 - y) * np.log(1 - prob)
    return -np.sum(log_likelihood)  # 取负值作为优化目标（最小化）

# 初始猜测值
beta_init = np.zeros(X.shape[1])

# 优化求解 MLE
result = minimize(neg_log_likelihood, beta_init, method='BFGS')

print("估计的参数值 (beta):", result.x)

