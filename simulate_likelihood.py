from daily_trad_vol import DailyTradVolRobust
import numpy as np
from particles import state_space_models as ssm
from particles import SMC
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def compute_logL_for_sy(sy, data, N=1000):
    model = DailyTradVolRobust(sigma_y=sy, c=1000)
    fk_model = ssm.Bootstrap(ssm=model, data=data)
    pf = SMC(fk=fk_model, N=N, resampling='stratified')
    pf.run()

    model_robust = DailyTradVolRobust(sigma_y=sy, c=5.14)
    fk_model_robust = ssm.Bootstrap(ssm=model_robust, data=data)
    pf_robust = SMC(fk=fk_model_robust, N=N, resampling='stratified')
    pf_robust.run()
    
    return (sy, pf.logLt, pf_robust.logLt)

def compute_logL_for_sx(sx, data, N=1000):
    model = DailyTradVolRobust(sigma_x=sx, c=1000)
    fk_model = ssm.Bootstrap(ssm=model, data=data)
    pf = SMC(fk=fk_model, N=N, resampling='stratified')
    pf.run()

    model_robust = DailyTradVolRobust(sigma_x=sx, c=5.14)
    fk_model_robust = ssm.Bootstrap(ssm=model_robust, data=data)
    pf_robust = SMC(fk=fk_model_robust, N=N, resampling='stratified')
    pf_robust.run()

    return (sx, pf.logLt, pf_robust.logLt)

def compute_logL_for_a(a, data, N=1000):
    model = DailyTradVolRobust(a=a, c=1000)
    fk_model = ssm.Bootstrap(ssm=model, data=data)
    pf = SMC(fk=fk_model, N=N, resampling='stratified')
    pf.run()

    model_robust = DailyTradVolRobust(a=a, c=5.14)
    fk_model_robust = ssm.Bootstrap(ssm=model_robust, data=data)
    pf_robust = SMC(fk=fk_model_robust, N=N, resampling='stratified')
    pf_robust.run()

    return (a, pf.logLt, pf_robust.logLt)

def compute_logL_for_b(b, data, N=1000):
    model = DailyTradVolRobust(b=b, c=1000)
    fk_model = ssm.Bootstrap(ssm=model, data=data)
    pf = SMC(fk=fk_model, N=N, resampling='stratified')
    pf.run()

    model_robust = DailyTradVolRobust(b=b, c=5.14)
    fk_model_robust = ssm.Bootstrap(ssm=model_robust, data=data)
    pf_robust = SMC(fk=fk_model_robust, N=N, resampling='stratified')
    pf_robust.run()

    return (b, pf.logLt, pf_robust.logLt)

def compute_logL_for_all(n, data): 
    """
    Compute log-likelihoods over grids for each parameter (as in fig. 10 & 11).
    1. sigma_y in [0.1, 1.4]
    2. sigma_x in [0.001, 0.4]
    3. a in [0.0, 0.1]
    4. b in [0.985, 0.9999]
    """
    grid_sy = np.linspace(0.1, 1.4, n)
    grid_sx = np.linspace(0.001, 0.4, n)
    grid_a = np.linspace(0.0, 0.1, n)
    grid_b = np.linspace(0.985, 0.9999, n)
    print("Starting computations...")
    results_sy = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_logL_for_sy)(sy, data) for sy in grid_sy
    )
    print("Finished sy computations.")
    results_sx = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_logL_for_sx)(sx, data) for sx in grid_sx
    )
    print("Finished sx computations.")
    results_a = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_logL_for_a)(a, data) for a in grid_a
    )
    print("Finished a computations.")
    results_b = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_logL_for_b)(b, data) for b in grid_b
    )
    print("Finished b computations.")
    return results_sy, results_sx, results_a, results_b

def plot_simulated_likelihood(results_sy, results_sx, results_a, results_b, type):

    sy_vals, sy_logL, sy_logL_robust = zip(*results_sy)
    sx_vals, sx_logL, sx_logL_robust = zip(*results_sx)
    a_vals, a_logL, a_logL_robust = zip(*results_a)
    b_vals, b_logL, b_logL_robust = zip(*results_b)

    fig, axes = plt.subplots(2, 4, figsize=(12, 12))

    axes[0, 0].plot(sy_vals, sy_logL, '-', color='tab:blue', linewidth=1)
    axes[0, 0].set_title(r'Standard: $\sigma_y$')
    axes[0, 0].set_ylabel('Log-Likelihood')
    axes[0, 0].set_xlabel(r'$\sigma_y$')

    axes[0, 1].plot(sx_vals, sx_logL, '-', color='tab:blue', linewidth=1)
    axes[0, 1].set_title(r'Standard: $\sigma_x$')
    axes[0, 1].set_xlabel(r'$\sigma_x$')

    axes[0, 2].plot(a_vals, a_logL, '-', color='tab:blue', linewidth=1)
    axes[0, 2].set_title(r'Standard: $a$')
    axes[0, 2].set_xlabel(r'$a$')

    axes[0, 3].plot(b_vals, b_logL, '-', color='tab:blue', linewidth=1)
    axes[0, 3].set_title(r'Standard: $b$')
    axes[0, 3].set_xlabel(r'$b$')

    axes[1, 0].plot(sy_vals, sy_logL_robust, '-', color='tab:orange', linewidth=1)
    axes[1, 0].set_title(r'Robust: $\sigma_y$')
    axes[1, 0].set_ylabel('Log-Likelihood')
    axes[1, 0].set_xlabel(r'$\sigma_y$')

    axes[1, 1].plot(sx_vals, sx_logL_robust, '-', color='tab:orange', linewidth=1)
    axes[1, 1].set_title(r'Robust: $\sigma_x$')
    axes[1, 1].set_xlabel(r'$\sigma_x$')

    axes[1, 2].plot(a_vals, a_logL_robust, '-', color='tab:orange', linewidth=1)
    axes[1, 2].set_title(r'Robust: $a$')
    axes[1, 2].set_xlabel(r'$a$')
    axes[1, 3].plot(b_vals, b_logL_robust, '-', color='tab:orange', linewidth=1)
    axes[1, 3].set_title(r'Robust: $b$')
    axes[1, 3].set_xlabel(r'$b$')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.suptitle(f'Simulated Likelihoods for {type}', fontsize=16, y=1.02)
    plt.show()