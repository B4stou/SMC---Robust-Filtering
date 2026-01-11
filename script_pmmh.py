"""
Shared models and functions for SMC particle filtering.
This module contains all dependencies needed for PMMH and other analyses.
"""

import numpy as np
from scipy.stats import norm
from numba import jit

from particles.core import FeynmanKac, SMC
import particles
particles.FeynmanKac = FeynmanKac

from particles import state_space_models as ssms
from particles import distributions as dists



class TradingVolumeModel(ssms.StateSpaceModel):
    """Trading volume state-space model.
    
    X_0 = a
    X_t = a + b * X_{t-1} + sigma_x * |X_{t-1}| * w_t, w_t ~ N(0,1)
    Y_t | X_t = x_t ~ N(x_t - sigma_y^2/2, sigma_y^2) univariate gaussian model comme section 3.2
    V_t = exp(Y_t)  
    """
    default_params = {'sigma_y': 0.3, 'a': 0.015, 'b': 0.998, 'sigma_x': 0.01}
    
    def PX0(self):
        return dists.Dirac(loc=self.a)
    
    def PX(self, t, xp):
        loc = self.a + self.b * xp
        return TradingVolumeTransition(loc=loc, scale=self.sigma_x, xp=xp)
    
    def PY(self, t, xp, x):
        # Y_t | X_t = x_t ~ N(x_t - sigma_y^2/2, sigma_y^2)
        mu = x - self.sigma_y**2 / 2
        return dists.Normal(loc=mu, scale=self.sigma_y)


class TradingVolumeTransition(dists.ProbDist):
    """Custom transition distribution for trading volume model."""
    def __init__(self, loc, scale, xp):
        self.loc = np.asarray(loc)
        self.scale = scale
        self.xp = np.asarray(xp)
        self.dim = 1
        self.dtype = np.float64
    
    def rvs(self, size=None):
        # When called from PX, size should match the number of particles (len(xp))
        # The loc and xp are arrays of the same length
        if size is None:
            size = len(self.loc) if hasattr(self.loc, '__len__') else 1
        # Generate random noise
        w = np.random.randn(size)
        # Handle both scalar and array cases
        if np.isscalar(self.loc):
            return self.loc + self.scale * np.abs(self.xp) * w
        else:
            # Ensure w has the right shape
            if len(w) != len(self.loc):
                w = np.random.randn(len(self.loc))
            return self.loc + self.scale * np.abs(self.xp) * w
    
    def logpdf(self, x):
        # Log density of N(loc, (scale * |xp|)^2)
        x = np.asarray(x)
        scale_abs = self.scale * np.abs(self.xp)
        scale_abs = np.maximum(scale_abs, 1e-10)  # Avoid division by zero
        if np.isscalar(x):
            return norm.logpdf(x, loc=self.loc, scale=scale_abs)
        else:
            return norm.logpdf(x, loc=self.loc, scale=scale_abs)
    
    def ppf(self, u):
        # Quantile function
        u = np.asarray(u)
        scale_abs = self.scale * np.abs(self.xp)
        scale_abs = np.maximum(scale_abs, 1e-10)
        if np.isscalar(u):
            return norm.ppf(u, loc=self.loc, scale=scale_abs)
        else:
            return norm.ppf(u, loc=self.loc, scale=scale_abs)


#  ROBUST GAUSSIAN DENSITY (JIT-compiled)
@jit(nopython=True, cache=True)
def robust_gaussian_density(y_t, x_t, sigma_y, mu_t, c=5.14):
    """Robust Gaussian density function (JIT-compiled for speed)."""
    mu_x = x_t - sigma_y**2 / 2
    delta = (mu_x - mu_t)**2 + 4 * c * sigma_y**2
    if delta < 0:
        return 1e-300
    y_minus = (mu_x + mu_t - np.sqrt(delta)) / 2
    y_plus = (mu_x + mu_t + np.sqrt(delta)) / 2
    # Compute Gaussian PDF manually for numba compatibility
    sqrt_2pi = 2.5066282746310002  # sqrt(2*pi)
    f_minus = np.exp(-0.5 * ((y_minus - mu_x) / sigma_y)**2) / (sigma_y * sqrt_2pi)
    f_plus = np.exp(-0.5 * ((y_plus - mu_x) / sigma_y)**2) / (sigma_y * sqrt_2pi)
    D_1 = f_minus * np.abs(y_minus - mu_t + 1e-10)**c
    D_2 = f_plus * np.abs(y_plus - mu_t + 1e-10)**c
    B = 1.0099
    if y_t < y_minus:
        density = B * D_1 * np.abs(y_t - mu_t + 1e-10)**(-c)
    elif y_t < y_plus:
        sqrt_2pi = 2.5066282746310002  # sqrt(2*pi)
        density = B * np.exp(-0.5 * ((y_t - mu_x) / sigma_y)**2) / (sigma_y * sqrt_2pi)
    else:
        density = B * D_2 * np.abs(y_t - mu_t + 1e-10)**(-c)
    return max(density, 1e-300)


# ROBUST BOOTSTRAP FILTER
class RobustBootstrap(ssms.Bootstrap):
    """Robust bootstrap filter using robust Gaussian density."""
    def __init__(self, ssm=None, data=None, c=5.14):
        super().__init__(ssm=ssm, data=data)
        self.c = c
    
    def logG(self, t, xp, x):
        """Robust weight function using robust Gaussian density."""
        sigma_y = self.ssm.sigma_y
        # x is always an array of particles
        x = np.asarray(x)
        mu_x = x - sigma_y**2 / 2
        mu_t = np.mean(mu_x)
        y_t = self.data[t]
        
        # Compute robust density for each particle
        log_densities = np.array([
            np.log(robust_gaussian_density(y_t, x_i, sigma_y, mu_t, self.c))
            for x_i in x
        ])
        
        return log_densities


# PARTICLE FILTERS USING PARTICLES PACKAGE 
def bootstrap_pf_particles(y, theta, N=1000):
    """Bootstrap (standard Gaussian) particle filter using particles package."""
    sigma_y, a, b, sigma_x = theta
    # Create model
    model = TradingVolumeModel(sigma_y=sigma_y, a=a, b=b, sigma_x=sigma_x)
    # Create bootstrap filter
    fk = ssms.Bootstrap(ssm=model, data=y)
    # Run SMC
    pf = SMC(fk=fk, N=N, store_history=True)
    pf.run()
    # Extract filtered states (mean of particles at each time)
    x_filtered = np.array([np.mean(pf.hist.X[t]) for t in range(len(y))])
    # Extract log-likelihood
    loglik = pf.summaries.logLts[-1] if pf.summaries is not None and hasattr(pf.summaries, 'logLts') else 0.0
    # Extract ESS history
    ess_history = np.array(pf.summaries.ESSs) if pf.summaries is not None else np.ones(len(y)) * N
    return x_filtered, loglik, ess_history

def robust_pf_particles(y, theta, N=1000, c=5.14):
    """Robust particle filter using particles package."""
    sigma_y, a, b, sigma_x = theta
    # Create model
    model = TradingVolumeModel(sigma_y=sigma_y, a=a, b=b, sigma_x=sigma_x)
    # Create robust bootstrap filter
    fk = RobustBootstrap(ssm=model, data=y, c=c)
    # Run SMC
    pf = SMC(fk=fk, N=N, store_history=True)
    pf.run()
    # Extract filtered states (mean of particles at each time)
    x_filtered = np.array([np.mean(pf.hist.X[t]) for t in range(len(y))])
    # Extract log-likelihood
    loglik = pf.summaries.logLts[-1] if pf.summaries is not None and hasattr(pf.summaries, 'logLts') else 0.0
    # Extract ESS history
    ess_history = np.array(pf.summaries.ESSs) if pf.summaries is not None else np.ones(len(y)) * N
    return x_filtered, loglik, ess_history

