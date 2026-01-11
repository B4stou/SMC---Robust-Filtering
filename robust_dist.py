from scipy import stats
from particles import distributions as dists
import numpy as np
import matplotlib.pyplot as plt

class RobustDist(dists.ProbDist):
    """
    Robustified Spherical Gaussian distribution with Gaussian center
    and polynomial tails (Eq 3.13 with p=1)
    """
    def __init__(self, loc, scale, c):
        self.loc = loc      
        self.scale = scale  
        self.c = c          
        
        sqrt_c = np.sqrt(c)
        phi = stats.norm.pdf(sqrt_c)
        Phi = stats.norm.cdf(sqrt_c)
        inv_B = 2 * Phi - 1 + (2 * sqrt_c / (c - 1)) * phi
        self.log_B = -np.log(inv_B)

    def logpdf(self, y):
    
        mu = self.loc
        sigma = self.scale
        safe_sigma = np.maximum(sigma, 1e-8)
        c = self.c
        sqrt_c = np.sqrt(c)
        
        
        z = (y - mu) / safe_sigma
        abs_z = np.abs(z)
        
        # Mask to identify central region vs tails
        mask_center = abs_z <= sqrt_c
        
        log_pdf = np.zeros_like(z)
        
        # --- 1. Central Part (Gaussian Behavior) ---
        # log(B) - log(sigma) - 0.5*log(2pi) - 0.5*z^2
        log_pdf[mask_center] = (
            self.log_B 
            - np.log(safe_sigma[mask_center]) 
            - 0.5 * np.log(2 * np.pi) 
            - 0.5 * (z[mask_center]**2)
        )
        
        # --- 2. Tail Part (Polynomial Behavior Eq 3.13) ---
        # log(B) - log(sigma) - 0.5*log(2pi) - c/2 - c*log(|z|/sqrt(c))
        log_pdf[~mask_center] = (
            self.log_B
            - np.log(safe_sigma[~mask_center])
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * c
            - c * np.log(abs_z[~mask_center] / sqrt_c)
        )
        
        log_pdf = np.nan_to_num(log_pdf, nan=-1e10, neginf=-1e10)
        return log_pdf
    
def get_robust_pdf(z, c):
    """
    Compute the PDF of the Robust Distribution at points z, given tail parameter c.
    Based on Proposition 4 of the paper (eq 3.13 with p = 1).
    """
    
    sqrt_c = np.sqrt(c)
    
    phi_sc = stats.norm.pdf(sqrt_c)
    Phi_sc = stats.norm.cdf(sqrt_c)
    inv_B = 2 * Phi_sc - 1 + (2 * sqrt_c / (c - 1)) * phi_sc
    B = 1 / inv_B
    

    abs_z = np.abs(z)
    pdf = np.zeros_like(z)
    
    # Masques
    mask_center = abs_z <= sqrt_c
    mask_tail = ~mask_center
    
    
    pdf[mask_center] = B * np.exp(-0.5 * z[mask_center]**2) / np.sqrt(2*np.pi)

    K = B / (np.sqrt(2*np.pi) * np.exp(c/2))
    pdf[mask_tail] = K * (abs_z[mask_tail] / sqrt_c)**(-c)
    
    return pdf

def plot_robust_vs_normal(c = 5.14):
    """
    Plot the Robust Distribution against the Standard Normal Distribution
    for a given tail parameter c.
    """
    
    z_values = np.linspace(-8, 8, 1000)
    c_param = c

    pdf_normal = stats.norm.pdf(z_values)

    pdf_robust = get_robust_pdf(z_values, c=c_param)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(z_values, pdf_normal, 'k--', label='Standard Gaussian', linewidth=2, alpha=0.6)
    ax1.plot(z_values, pdf_robust, 'r-', label=f'Robust (c={c_param})', linewidth=2)
    ax1.set_title('Comparison - Linear Scale (Center of the Distribution)')
    ax1.set_xlabel('z (standard deviations)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4, 4) 

    ax2.plot(z_values, pdf_normal, 'k--', label='Standard Gaussian  ', linewidth=2, alpha=0.6)
    ax2.plot(z_values, pdf_robust, 'r-', label=f'Robust (c={c_param})', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_title('Comparison - Logarithmic Scale (Tails)')
    ax2.set_xlabel('z (standard deviations)')
    ax2.set_ylabel('Log-Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim(1e-6, 1) 

    sqrt_c = np.sqrt(c_param)
    ax2.axvline(x=sqrt_c, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=-sqrt_c, color='gray', linestyle=':', alpha=0.5)
    ax2.text(sqrt_c + 0.2, 1e-1, r'$\sqrt{c}$ (Transition)', color='gray')

    plt.tight_layout()
    plt.show()