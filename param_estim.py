import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import particles
from particles import state_space_models as ssm
from particles import distributions as dists
from particles.collectors import Moments
from particles import SMC

from scipy import stats
from scipy.optimize import minimize

from joblib import Parallel, delayed 
import time

from daily_trad_vol import DailyTradVolRobust
from robust_dist import RobustDist

def get_neg_loglik_safe(params, data, c_robust):
    """
    Compute negative log-likelihood with parameter constraints and error handling
    """
    try:
        # Constraints on parameters
        sigma_y = np.exp(params[0]) 
        a = np.abs(params[1])
        b = np.tanh(params[2])
        sigma_x = np.exp(params[3])
        
        if sigma_x > 1.0 or sigma_y > 2.0:
            return 1e12 
        
        model = DailyTradVolRobust(
            data=data, sigma_y=sigma_y, a=a, b=b, sigma_x=sigma_x, c=c_robust
        )
        
        fk = ssm.Bootstrap(ssm=model, data=data)
        pf = SMC(fk=fk, N=500)
        pf.run()
        
        if np.isnan(pf.logLt) or np.isinf(pf.logLt):
            return 1e12
            
        return -pf.logLt

    except Exception:
        return 1e12 


def generate_random_x0():
    """
    Generate a realistic random initial parameter vector.
    """
    # sigma_y : log-uniform between 0.1 and 1.0

    log_sy = np.random.uniform(np.log(0.1), np.log(1.0))
    
    # a : uniform between 0.0 and 0.1
    a_val = np.random.uniform(0.0, 0.1)
    
    # b : very persistent, between 0.90 and 0.999
    b_val = np.random.uniform(0.90, 0.999)
    atanh_b = np.arctanh(b_val)
    
    # sigma_x : log-uniform between 0.001 and 0.9
    log_sx = np.random.uniform(np.log(0.001), np.log(0.9))
    
    return [log_sy, a_val, atanh_b, log_sx]


def run_one_optimization(i, data, c_robust):
    """Run a full optimization from a random starting point."""
    np.random.seed(i)
    
    x0 = generate_random_x0()
    print(f"[Worker {i}]  Starting...", flush=True)
    # Local optimizations
    res_robpf = minimize(
        get_neg_loglik_safe, 
        x0, 
        args=(data, c_robust),
        method='Nelder-Mead',
        options={'maxiter': 60, 'disp': False} 
    )

    res_pf = minimize(
        get_neg_loglik_safe, 
        x0, 
        args=(data, 1000),
        method='Nelder-Mead',
        options={'maxiter': 60, 'disp': False}
    )
    
    print(f"[Worker {i}]  Terminé.", flush=True)
    
    return {
        'id': i,
        'success_pf': res_pf.success,
        'score_pf': res_pf.fun,
        'params_pf': res_pf.x,
        'success_robpf': res_robpf.success,
        'score_robpf': res_robpf.fun,
        'params_robpf': res_robpf.x,
    }

def run_parallel_optimizations(data_train, N=48, n_jobs=-1, c_robust=5.14):
    """
    Run N optimizations in parallel on the training data.
    """

    print(f"Lancement de {N} optimisations en PARALLÈLE...")
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_one_optimization)(i, data_train, c_robust) 
    for i in range(N)
    )

    end_time = time.time()
    print(f"End in {end_time - start_time:.1f} seconds.")

    return results

def process_results(results):
    """
    Process the results from the parallel optimizations into a DataFrame.
    """
    data_list = []
    for r in results:
        # RobPF
        if r['score_robpf'] < 1e13:
            p = r['params_robpf']
            data_list.append({
                'Method': 'Robust (c=5.14)',
                'LogLik': -r['score_robpf'],
                'sigma_y': np.exp(p[0]),
                'a': np.abs(p[1]),
                'b': np.tanh(p[2]),
                'sigma_x': np.exp(p[3]),
                'ID': r['id']
            })
            
        # PF
        if r['score_pf'] < 1e13:
            p = r['params_pf']
            data_list.append({
                'Method': 'Standard (c=1000)',
                'LogLik': -r['score_pf'],
                'sigma_y': np.exp(p[0]),
                'a': np.abs(p[1]),
                'b': np.tanh(p[2]),
                'sigma_x': np.exp(p[3]),
                'ID': r['id']
            })

    df_results = pd.DataFrame(data_list)

    return df_results

def show_results(df_results):
    """
    Print the best results for each method.
    """
    res ={}
    for method in df_results['Method'].unique():
        best_run = df_results[df_results['Method'] == method].sort_values('LogLik', ascending=False).iloc[0]
        
        print(f"--- {method} ---")
        print(f"   Log-Likelihood : {best_run['LogLik']:.2f}")
        print(f"   sigma_y        : {best_run['sigma_y']:.4f}")
        print(f"   b              : {best_run['b']:.4f}")
        print(f"   sigma_x        : {best_run['sigma_x']:.4f}")
        print(f"   a              : {best_run['a']:.4f}")
        print("-" * 30)
        res[method] = best_run
    return res
    

def plot_boxplots(df_results):
    """
    Plot boxplots of parameter estimates for each method, as in the Appendix.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    param_names = ['sigma_y', 'a', 'b', 'sigma_x']
    best_params = {}

    for method in df_results['Method'].unique():
        best_run = df_results[df_results['Method'] == method].sort_values('LogLik', ascending=False).iloc[0]
        best_params[method] = {
            'sigma_y': best_run['sigma_y'],
            'a': best_run['a'],
            'b': best_run['b'],
            'sigma_x': best_run['sigma_x']
        }

    for ax, param in zip(axes.flatten(), param_names):
        sns.boxplot(x='Method', y=param, data=df_results, ax=ax, hue='Method', palette='Set2')
        ax.set_title(f"Distribution de l'estimation de {param}")
        
        for method, params in best_params.items():
            ax.scatter(method, params[param], color='red', s=100, zorder=5, label='Best Fit' if param == param_names[0] else "")
        
        if param == param_names[0]:
            ax.legend()
    plt.tight_layout()
    plt.show()
