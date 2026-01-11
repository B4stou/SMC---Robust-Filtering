import numpy as np
from particles import state_space_models as ssm
from particles import distributions as dists
from particles.collectors import Moments
from particles import SMC
from scipy import stats
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt

import robust_dist
from robust_dist import RobustDist

class DailyTradVol(ssm.StateSpaceModel):
    """
    Daily trading volume model (section 4.4)
    """
    # Specified in section 4.4
    default_params = {'sigma_y': 0.3, 'sigma_x': 0.01, 'a': 0.015, 'b': 0.998}
    
    def PX0(self):
        return dists.Normal(loc=(self.a/(1-self.b)), scale=self.sigma_x)
    
    def PX(self, t, xp):
        # x_t | x_{t-1} eq. 4.6
        return dists.Normal(loc=self.a+self.b*xp, scale=self.sigma_x*abs(xp))
    
    def PY(self, t, xp, x):
        # y_t | x_t eq. 4.5
        return dists.Normal(loc=x-(self.sigma_y)**2/2, scale=self.sigma_y)

class DailyTradVolRobust(ssm.StateSpaceModel):
    """
    Daily trading volume model with robust observation distribution.
    """
    # Specified in section 4.4
    default_params = {'sigma_y': 0.3, 'sigma_x': 0.01, 'a': 0.015, 'b': 0.998, 'c':5.14} 
    
    def PX0(self):
        
        return dists.Normal(loc=(self.a/(1-self.b)), scale=self.sigma_x)
    
    def PX(self, t, xp):
        # x_t | x_{t-1} (eq. 4.6)
        return dists.Normal(loc=self.a+self.b*xp, scale=self.sigma_x*abs(xp)+1e-6)
    
    def PY(self, t, xp, x):
        # y_t | x_t (eq. 4.5) with robust distribution
        mu = x - (self.sigma_y)**2/2
        scale_vec = np.full_like(x, self.sigma_y)

        return RobustDist(loc=mu, scale=scale_vec, c=self.c)
    
def contaminate_trading_volume(y_clean, x_clean, sigma_y, rate=0.05, eta=14):
    """
    Contaminate trading volume data by adding uniform noise to a fraction of observations,
    as described in figure 11
    """
    y_cont = y_clean.copy()
    T = len(y_clean)
    
    for t in range(T):
        
        if np.random.rand() < rate:
            
            mu_t = x_clean[t] - (sigma_y**2) / 2
            radius = np.abs(y_clean[t] - mu_t)
            u_t = np.random.uniform(-radius, radius)
            y_cont[t] = y_clean[t] + eta * u_t
            
    return y_cont

def get_ess_ratio(data, c, N=1000, thres=0.01):
    

    model = DailyTradVolRobust(c=c)
    fk_model = ssm.Bootstrap(ssm=model, data=data)
    pf = SMC(fk=fk_model, N=N, resampling='stratified', collect=[Moments()], store_history=True)
    pf.run()
    ess_ratio = np.array(pf.summaries.ESSs) / N
    prop_below_thres = np.mean(ess_ratio < thres)

    return ess_ratio, prop_below_thres

def get_ess_prop_eta(data, true_states, threshold=0.01):
    threshold = 0.01
    grid_eta = np.linspace(-100,100, 50)
    prop = []
    prop_rob = []
    for eta in tqdm(grid_eta, desc="Processing eta values"):
        contaminated_data = contaminate_trading_volume(data, true_states, sigma_y=0.3, rate=0.05, eta=eta)
        fk_model_std = ssm.Bootstrap(ssm=DailyTradVol(), data=contaminated_data)
        pf_std = SMC(fk=fk_model_std, N=1000, resampling='stratified', collect=[Moments()], store_history=True)
        pf_std.run()
        ess_ratio_std = np.array(pf_std.summaries.ESSs) / 1000
        proportion_below_threshold = np.mean(ess_ratio_std < threshold) * 100
        prop.append(proportion_below_threshold)
        fk_model_rob = ssm.Bootstrap(ssm=DailyTradVolRobust(c=5.14), data=contaminated_data)
        pf_rob = SMC(fk=fk_model_rob, N=1000, resampling='stratified', collect=[Moments()], store_history=True)
        pf_rob.run()
        ess_ratio_rob = np.array(pf_rob.summaries.ESSs) / 1000
        proportion_below_threshold_rob = np.mean(ess_ratio_rob < threshold) * 100
        prop_rob.append(proportion_below_threshold_rob)
    
    return grid_eta, prop, prop_rob



##################################
###### REAL DATA FORECASTING #####
##################################


def get_forecasted_volumes(param, data) :
    """
    Get one-day-ahead forecasted trading volumes using the given parameters.
    We follow the method given at the end of 4.4.3
    """
    model = DailyTradVolRobust(data=data, **param)
    fk = ssm.Bootstrap(ssm=model, data=data)
    pf = SMC(fk=fk, N=1000, store_history=True)
    pf.run()

    X_matrix = np.array(pf.hist.X) # Particles at time t before weighting for all times
    T, N = X_matrix.shape
    sigma_y = param['sigma_y']
    
    noise_matrix = np.random.normal(loc=0.0, scale=sigma_y, size=(T, N))
    
    V_matrix = X_matrix - (sigma_y**2)/2 + noise_matrix # Eq 4.5
    forecast_one_day_ahead = np.mean(V_matrix, axis=1) # Average 
    
    predicted_volumes = np.exp(forecast_one_day_ahead)
    volume_actual = np.exp(data)

    r2 = r2_score(volume_actual, predicted_volumes)
    mse = np.mean((volume_actual - predicted_volumes)**2)
    mae = np.mean(np.abs(volume_actual - predicted_volumes))
    
    return {'forecast_one_day_ahead': forecast_one_day_ahead, 'predicted_volumes': predicted_volumes, 'r2': r2, 'mse': mse, 'mae': mae}

def plot_results_comparison(data_test, result_pf, result_robpf):
   
    print("=" * 60)
    print("OUT-OF-SAMPLE PREDICTION METRICS COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'Standard PF':<20} {'Robust PF':<20}")
    print("-" * 60)
    print(f"{'R² Score':<20} {result_pf['r2']:<20.4f} {result_robpf['r2']:<20.4f}")
    print(f"{'MSE':<20} {result_pf['mse']:<20.2f} {result_robpf['mse']:<20.2f}")
    print(f"{'MAE':<20} {result_pf['mae']:<20.2f} {result_robpf['mae']:<20.2f}")
    print("=" * 60)
    print(f"\nImprovement with Robust PF:")
    print(f"  - R² improvement: {(result_robpf['r2'] - result_pf['r2']):.4f} ({((result_robpf['r2'] - result_pf['r2'])/result_pf['r2']*100):.1f}%)")
    print(f"  - MSE reduction: {(result_pf['mse'] - result_robpf['mse']):.2f} ({((result_pf['mse'] - result_robpf['mse'])/result_pf['mse']*100):.1f}%)")
    print(f"  - MAE reduction: {(result_pf['mae'] - result_robpf['mae']):.2f} ({((result_pf['mae'] - result_robpf['mae'])/result_pf['mae']*100):.1f}%)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # PF
    axes[0].plot(np.exp(data_test), color='black', alpha=0.5, linewidth=0.8, label='True Volume')
    axes[0].plot(result_pf['predicted_volumes'], color='blue', linewidth=1.2, label='PF Predicted')
    axes[0].set_title('Predicted Trading Volume (PF)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Volume')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RobPF
    axes[1].plot(np.exp(data_test), color='black', alpha=0.5, linewidth=0.8, label='True Volume')
    axes[1].plot(result_robpf['predicted_volumes'], color='red', linewidth=1.2, label='RobPF Predicted')
    axes[1].set_title('Predicted Trading Volume (RobPF)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Volume')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_result_regression_comparison(data_test, result_pf, result_robpf):

    burn_in = 50 

    # PF
    Y_actual_pf = np.exp(data_test)[burn_in:]
    Y_pred_pf = result_pf['predicted_volumes'][burn_in:]
    slope_pf, intercept_pf, r_value_pf, p_value_pf, std_err_pf = stats.linregress(Y_pred_pf, Y_actual_pf)


    # RobPF
    Y_actual_robpf = np.exp(data_test)[burn_in:]
    Y_pred_robpf = result_robpf['predicted_volumes'][burn_in:]
    slope_robpf, intercept_robpf, r_value_robpf, p_value_robpf, std_err_robpf = stats.linregress(Y_pred_robpf, Y_actual_robpf)


    print("=" * 60)
    print("REGRESSION RESULTS (Mincer-Zarnowitz)")
    print("=" * 60)
    print("\nSTANDARD PF:")
    print(f"Equation : Actual = {intercept_pf:.4f} + {slope_pf:.4f} * Predicted")
    print(f"Slope (Beta)  : {slope_pf:.4f}  (Ideal target = 1.0)")
    print(f"Bias (Alpha) : {intercept_pf:.4f}  (Ideal target = 0.0)")

    print("\nROBUST PF:")
    print(f"Equation : Actual = {intercept_robpf:.4f} + {slope_robpf:.4f} * Predicted")
    print(f"Slope (Beta)  : {slope_robpf:.4f}  (Ideal target = 1.0)")
    print(f"Bias (Alpha) : {intercept_robpf:.4f}  (Ideal target = 0.0)")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PF
    axes[0].scatter(Y_pred_pf, Y_actual_pf, alpha=0.1, color='blue', s=10, label='Data')
    x_range_pf = np.linspace(Y_pred_pf.min(), Y_pred_pf.max(), 100)
    axes[0].plot(x_range_pf, intercept_pf + slope_pf * x_range_pf, 'r-', linewidth=2, label=f'Regression (Slope={slope_pf:.2f})')
    axes[0].plot([Y_actual_pf.min(), Y_actual_pf.max()], [Y_actual_pf.min(), Y_actual_pf.max()], 'k--', label='Ideal (y=x)')
    axes[0].set_xlabel("Predicted Volume", fontsize=11)
    axes[0].set_ylabel("Actual Volume", fontsize=11)
    axes[0].set_title(f"Standard PF - Regression: Actual vs Predicted", fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RobPF
    axes[1].scatter(Y_pred_robpf, Y_actual_robpf, alpha=0.1, color='red', s=10, label='Data')
    x_range_robpf = np.linspace(Y_pred_robpf.min(), Y_pred_robpf.max(), 100)
    axes[1].plot(x_range_robpf, intercept_robpf + slope_robpf * x_range_robpf, 'r-', linewidth=2, label=f'Regression (Slope={slope_robpf:.2f})')
    axes[1].plot([Y_actual_robpf.min(), Y_actual_robpf.max()], [Y_actual_robpf.min(), Y_actual_robpf.max()], 'k--', label='Ideal (y=x)')
    axes[1].set_xlabel("Predicted Volume", fontsize=11)
    axes[1].set_ylabel("Actual Volume", fontsize=11)
    axes[1].set_title(f"Robust PF - Regression: Actual vs Predicted", fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
