import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any

def aggregate_telemetry(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Aggregates user telemetry data to calculate trials and conversions for each variant.
    
    Parameters:
        df (pd.DataFrame): Telemetry DataFrame with 'assigned_variant' and 'profile_completed' columns.
        
    Returns:
        Dict[str, Dict[str, int]]: Dictionary mapping each variant ('A', 'B') to its trials and conversions count:
                                   {
                                       'A': {'trials': int, 'conversions': int},
                                       'B': {'trials': int, 'conversions': int}
                                   }
    """
    # Group by assigned variant and compute conversions (sum) and trials (count)
    agg = df.groupby('assigned_variant').agg(
        trials=('profile_completed', 'count'),
        conversions=('profile_completed', 'sum')
    ).to_dict(orient='index')
    
    # Ensure both 'A' and 'B' are present with 0s if they don't exist in data
    for variant in ['A', 'B']:
        if variant not in agg:
            agg[variant] = {'trials': 0, 'conversions': 0}
            
    return agg

def compute_posterior_sampling(
    metrics: Dict[str, Dict[str, int]], 
    n_simulations: int = 100000, 
    prior_alpha: float = 1.0, 
    prior_beta: float = 1.0,
    seed: int = None
) -> Dict[str, Any]:
    """
    Computes conjugate beta posterior updates and samples from the posteriors using scipy.stats.beta.
    
    Parameters:
        metrics (Dict[str, Dict[str, int]]): Aggregated trials and conversions.
        n_simulations (int): Number of posterior Monte Carlo samples to draw.
        prior_alpha (float): Alpha parameter for Beta prior (default 1.0 for uniform prior).
        prior_beta (float): Beta parameter for Beta prior (default 1.0 for uniform prior).
        seed (int, optional): Random seed for reproducibility of samples.
        
    Returns:
        Dict[str, Any]: Dictionary containing posterior parameters, posterior samples, and key Bayesian metrics:
                        - 'posteriors': updated alpha and beta values for 'A' and 'B'.
                        - 'samples_A': np.ndarray of shape (n_simulations,) representing posterior conversion rate samples for A.
                        - 'samples_B': np.ndarray of shape (n_simulations,) representing posterior conversion rate samples for B.
                        - 'prob_B_gt_A': Float, probability that variant B is better than variant A.
                        - 'expected_lift': Float, expected relative improvement of B over A.
    """
    # Extract trials and conversions
    trials_a = metrics['A']['trials']
    conv_a = metrics['A']['conversions']
    
    trials_b = metrics['B']['trials']
    conv_b = metrics['B']['conversions']
    
    # Conjugate Beta-Binomial update: 
    # posterior_alpha = prior_alpha + conversions
    # posterior_beta = prior_beta + (trials - conversions)
    post_alpha_a = prior_alpha + conv_a
    post_beta_a = prior_beta + (trials_a - conv_a)
    
    post_alpha_b = prior_alpha + conv_b
    post_beta_b = prior_beta + (trials_b - conv_b)
    
    # Set seed for reproducible SciPy sampling if provided
    random_state = np.random.default_rng(seed)
    
    # Sample from Beta posterior using scipy.stats.beta
    samples_a = stats.beta.rvs(post_alpha_a, post_beta_a, size=n_simulations, random_state=random_state)
    samples_b = stats.beta.rvs(post_alpha_b, post_beta_b, size=n_simulations, random_state=random_state)
    
    # Calculate probability that B > A
    prob_b_gt_a = float(np.mean(samples_b > samples_a))
    
    # Calculate relative lift (expected B / expected A - 1)
    mean_a = np.mean(samples_a)
    mean_b = np.mean(samples_b)
    expected_lift = float((mean_b - mean_a) / mean_a) if mean_a > 0 else 0.0
    
    # Calculate 95% equal-tailed credible intervals (2.5th to 97.5th percentiles)
    ci_a = tuple(map(float, np.percentile(samples_a, [2.5, 97.5])))
    ci_b = tuple(map(float, np.percentile(samples_b, [2.5, 97.5])))
    
    lift_samples = (samples_b - samples_a) / samples_a
    ci_lift = tuple(map(float, np.percentile(lift_samples, [2.5, 97.5])))
    
    # Calculate Value-at-Risk (VaR) at 5th percentile of the relative lift distribution
    lift_var_5pct = float(np.percentile(lift_samples, 5))
    
    return {
        'posteriors': {
            'A': {'alpha': post_alpha_a, 'beta': post_beta_a},
            'B': {'alpha': post_alpha_b, 'beta': post_beta_b}
        },
        'samples_A': samples_a,
        'samples_B': samples_b,
        'prob_B_gt_A': prob_b_gt_a,
        'expected_lift': expected_lift,
        'ci_A': ci_a,
        'ci_B': ci_b,
        'ci_lift': ci_lift,
        'lift_var_5pct': lift_var_5pct
    }
