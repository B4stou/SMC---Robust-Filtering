import numpy as np
from particles import state_space_models as ssm
from particles import distributions as dists
from particles.collectors import Moments
from particles import SMC
from scipy import stats

import robust_dist
from robust_dist import RobustDist



class StochVol(ssm.StateSpaceModel):
    """
    Discrete short-term interest rate process with stochastic volatily (Section 4.3).
    Non-robust (Standard).
    """
    def __init__(self, data=None, alpha=0.002, beta=-0.001, gamma=0.9,
                 a=-0.05, b=0.99, sigma=0.12):
        self.data = data # Here y_t depens on y_{t-1}
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.b = b
        self.sigma = sigma

    def PX0(self):
        stat_mean = self.a / (1 - self.b)
        stat_std = self.sigma / np.sqrt(1 - self.b**2)
        return dists.Normal(loc=stat_mean, scale=stat_std)

    def PX(self, t, xp):
        
        return dists.Normal(loc=self.a + self.b*xp, scale=self.sigma)

    def PY(self, t, xp, x):
                
        if t == 0:
            return dists.Normal(loc=np.zeros_like(x), scale=0.1)
        
        y_prev = self.data[t-1]
        loc = y_prev + self.alpha + self.beta * y_prev
        
        scale = np.exp(x/2) * (np.abs(y_prev)**self.gamma)
        
        # To avoid scale=0 which causes issues in Normal distribution
        return dists.Normal(loc=loc, scale=scale + 1e-8)

    
    def simulate(self, T, x0=None, y0=0.0):
        if x0 is None:
            x0 = self.PX0().rvs(size=1)
        
        X = np.zeros(T)
        Y = np.zeros(T)
        
        X[0] = x0
        Y[0] = y0

        for t in range(1, T):
            # Transition X_t
            X[t] = self.PX(t, X[t-1]).rvs()
            
            # Observation Y_t (depends on X_t and Y_{t-1})
            y_prev = Y[t-1]
            loc = y_prev + self.alpha + self.beta * y_prev
            scale = np.exp(X[t]/2) * (np.abs(y_prev)**self.gamma)
            Y[t] = loc + scale * np.random.randn()

        return X, Y
    

class RobustStochVol(ssm.StateSpaceModel):
    """
    Discrete short-term interest rate process with stochastic volatily (Section 4.3).
    Robustified with RobustDist for observations.
    """
    default_params = {
        'alpha': 0.002, 'beta': -0.001, 'gamma': 0.9, 
        'a': -0.05, 'b': 0.99, 'sigma': 0.12,
        'c': 5.14 # RobustDist tail parameter, from paper (Table 1, with alpha =0.01)
    }

    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.data = data 

    def PX0(self):
        stat_mean = self.a / (1 - self.b)
        stat_std = self.sigma / np.sqrt(1 - self.b**2)
        return dists.Normal(loc=stat_mean, scale=stat_std)
        
    def PX(self, t, xp):
        
        return dists.Normal(loc=self.a + self.b * xp, scale=self.sigma)

    def PY(self, t, xp, x):
        
        if t == 0:
            return dists.Normal(loc=np.zeros_like(x), scale=0.1)
        else:
            y_prev = self.data[t-1]
        
        mu_t = y_prev + self.alpha + self.beta * y_prev
        sigma_val = np.exp(x / 2.0) * (np.abs(y_prev)**self.gamma)
        return RobustDist(loc=mu_t, scale=sigma_val, c=self.c)
    
def smoothing(data, states, N = 1000, n = 200, model = None) :
    """
    Perform smoothing on the Stochastic Volatility model
    """
    if model :
        model_Stoch_vol = model
    else :
        model_Stoch_vol = StochVol(data=data)
    fk_model_sv = ssm.Bootstrap(ssm=model_Stoch_vol, data=data)
    pf_sv = SMC(fk=fk_model_sv, N=N, resampling='stratified', collect=[Moments()], store_history=True)
    pf_sv.run()

    smooth_traj_stochvol_1 = pf_sv.hist.backward_sampling_ON2(n)
    smooth_traj_stochvol = [[s[i] for s in smooth_traj_stochvol_1] for i in range(n)]

    upper_stochvol = np.percentile(smooth_traj_stochvol, 95, axis=0)
    lower_stochvol = np.percentile(smooth_traj_stochvol, 5, axis=0)

    true_values = np.array(states)
    inside_interval = np.sum((true_values >= lower_stochvol) & (true_values <= upper_stochvol))
    percentage = (inside_interval / len(true_values)) * 100

    return smooth_traj_stochvol, upper_stochvol, lower_stochvol, percentage, model_Stoch_vol

def robust_smoothing(data, states, N = 1000, n = 200, c=5.14, model = None) :
    """
    Perform smoothing on the Robustified Stochastic Volatility model
    """
    if model :
        model_Robust_Stoch_vol = model
    else :
        model_Robust_Stoch_vol = RobustStochVol(data=data, c=c)
    fk_model_robust_sv = ssm.Bootstrap(ssm=model_Robust_Stoch_vol, data=data)
    pf_robust_sv = SMC(fk=fk_model_robust_sv, N=N, resampling='stratified', collect=[Moments()], store_history=True)
    pf_robust_sv.run()

    smooth_traj_robust_stochvol_1 = pf_robust_sv.hist.backward_sampling_ON2(n)
    smooth_traj_robust_stochvol = [[s[i] for s in smooth_traj_robust_stochvol_1] for i in range(n)]

    upper_robust_stochvol = np.percentile(smooth_traj_robust_stochvol, 95, axis=0)
    lower_robust_stochvol = np.percentile(smooth_traj_robust_stochvol, 5, axis=0)

    true_values = np.array(states)
    inside_interval = np.sum((true_values >= lower_robust_stochvol) & (true_values <= upper_robust_stochvol))
    percentage = (inside_interval / len(true_values)) * 100

    return smooth_traj_robust_stochvol, upper_robust_stochvol, lower_robust_stochvol, percentage, model_Robust_Stoch_vol

def generate_contaminated_data(y_clean, alpha, beta, contamination_rate=0.05, eta=10):
    """
    Generate contaminated observations based on the clean data.
    Uniform continuous contamination model, as in 2.2.2
    """
    y_cont = y_clean.copy()
    T = len(y_clean)
    
    for t in range(T):
        if np.random.rand() < contamination_rate:
            
            if t == 0:
                y_prev = 0.0
            else:
                y_prev = y_clean[t-1] 

            mu_t = y_prev + alpha + beta * y_prev
            radius = np.abs(y_clean[t] - mu_t)
            v = np.random.choice([-1.0, 1.0])
            r = radius * np.random.rand()
            u = r * v
            y_cont[t] = y_clean[t] + eta * u
            
    return y_cont