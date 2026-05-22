import numpy as np
import pandas as pd

def generate_user_logs(n_samples: int, seed: int = None, p_a: float = 0.30, p_b: float = 0.35) -> pd.DataFrame:
    """
    Generates simulated user platform telemetry data for an A/B testing scenario.
    
    Parameters:
        n_samples (int): The number of user logs to generate.
        seed (int, optional): Random seed for reproducibility.
        p_a (float): True conversion probability (profile completion rate) for variant A.
        p_b (float): True conversion probability (profile completion rate) for variant B.
        
    Returns:
        pd.DataFrame: DataFrame containing simulated telemetry data with columns:
                      - user_id (str)
                      - assigned_variant (str: 'A' or 'B')
                      - profile_completed (int: 0 or 1)
    """
    rng = np.random.default_rng(seed)
    
    # Generate unique user IDs
    user_ids = [f"user_{i:06d}" for i in range(1, n_samples + 1)]
    
    # Assign variants (A or B) using a binomial distribution (50/50 allocation)
    variant_indicator = rng.binomial(n=1, p=0.5, size=n_samples)
    assigned_variants = np.where(variant_indicator == 1, 'B', 'A')
    
    # Initialize profile completed outcomes
    profile_completed = np.zeros(n_samples, dtype=int)
    
    # Identify indices for variant A and B
    mask_a = (assigned_variants == 'A')
    mask_b = (assigned_variants == 'B')
    
    n_a = np.sum(mask_a)
    n_b = np.sum(mask_b)
    
    # Draw from binomial distribution with probability p_a/p_b for respective variants
    if n_a > 0:
        profile_completed[mask_a] = rng.binomial(n=1, p=p_a, size=n_a)
    if n_b > 0:
        profile_completed[mask_b] = rng.binomial(n=1, p=p_b, size=n_b)
        
    # Construct telemetry DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'assigned_variant': assigned_variants,
        'profile_completed': profile_completed
    })
    
    return df
