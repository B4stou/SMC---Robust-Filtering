# **SMC Projet : Robust Filtering**

___

*Alice Lastate, Bastien Lecomte & Imane Mokhtatif*

___

**Based on the following paper :** Laurent E. Calvet, Veronika Czellar & Elvezio Ronchetti (2015) Robust
Filtering, Journal of the American Statistical Association, 110:512, 1591-1606, DOI:
10.1080/01621459.2014.983520 


This project implements robust and bootstrap particle filters for gas trading volume data analysis, along with parameter estimation methods. The project compares standard and robust particle filtering approaches on simulated (clean and contaminated) and real gas trading volume data.

## Project Structure

### Main tool
**`main.ipynb`**
- This Jupyter Notebook orchestrates the simulations, trains the models, and generates the final figures presented in the slides. All the results are avaibable by doing a run-all.

### Stochastic Volatility model
**`stochvol.py`**
- Contains all the method relating on the part 4.3 of the paper (at the beginning of the notebook)

### Robust distribution
**`robus_dist.py`**
- Contains the definition of the distribution we use in stochvol and daily_trad_vol, and a plotting function.

### Daily Trad Vol
**`daily_trad_vol.py` / `simulate_likelihood.py`**
- Contains all the methods (model, prediction function, results plotting...) we need to reproduce the results of part 4.4 (with param_estim) 

### Parameter Estimation
**`param_estim.py`**
- Parameter estimation by Maximum Likelihood Estimation (MLE), comparing :
  - Standard particle filter likelihood estimation
  - Robust particle filter likelihood estimation
- This script contains the iteration loop for multiple starting points of Downhill Simplex algo, and some helper functions to plot things.

### Bayesian estimation : PMMH (Particle Marginal Metropolis-Hastings)
**`script_pmmh.py` / `Bonus_PMMH.ipynb`**
- Parameter posterior estimation through Particle Marginal Metropolis-Hastings (PMMH)


