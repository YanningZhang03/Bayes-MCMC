import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import time
import arviz as az

df = pd.read_csv('bayes_dataset.csv')
x = df['x'].values.astype(float)
y = df['y'].values.astype(int)
n = len(y)
MLE_beta0 = 1.3862
MLE_beta1 = -0.1880

mu_0, mu_1 = 0, 0
sigma_0, sigma_1 = 100, 100

def log_posterior(beta):
    beta0, beta1 = beta
    eta = beta0 + beta1 * x
    Phi = np.clip(norm.cdf(eta), 1e-6, 1-1e-6)
    ll = np.sum(y * np.log(Phi) + (1 - y) * np.log(1 - Phi))
    lp = -0.5*((beta0 - mu_0)**2 / sigma_0**2 + (beta1 - mu_1)**2 / sigma_1**2)
    return ll + lp

def grad_log_posterior(beta):
    beta0, beta1 = beta
    eta = beta0 + beta1 * x
    Phi = np.clip(norm.cdf(eta), 1e-6, 1-1e-6)
    phi = norm.pdf(eta)
    g0 = np.sum((y - Phi) * phi / (Phi * (1 - Phi))) - (beta0 - mu_0) / sigma_0**2
    g1 = np.sum((y - Phi) * phi * x / (Phi * (1 - Phi))) - (beta1 - mu_1) / sigma_1**2
    return np.array([g0, g1])

def log_q(b_from, b_to, grad_from, eps):
    mean = b_from + 0.5 * eps**2 * grad_from
    d = b_to - mean
    return -0.5 * np.dot(d, d) / (eps**2)

n_samples = 20000
burnin = 5000
epsilon = 0.03

samples = np.zeros((n_samples, 2))
beta_curr = np.array([0.0, 0.0])

start = time.time()
for t in range(n_samples):
    grad_curr = grad_log_posterior(beta_curr)
    noise = np.random.normal(size=2)
    beta_prop = beta_curr + 0.5 * epsilon**2 * grad_curr + epsilon * noise
    logq_f = log_q(beta_curr, beta_prop, grad_curr, epsilon)
    grad_prop = grad_log_posterior(beta_prop)
    logq_b = log_q(beta_prop, beta_curr, grad_prop, epsilon)
    log_alpha = log_posterior(beta_prop) - log_posterior(beta_curr) + logq_b - logq_f
    if np.log(np.random.rand()) < log_alpha:
        beta_curr = beta_prop
    samples[t] = beta_curr
end = time.time()

print(f"Sampling time: {end-start:.2f} s")

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
