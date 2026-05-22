import numpy as np
from src.data_generation import generate_user_logs
from src.bayesian_engine import aggregate_telemetry, compute_posterior_sampling

def run_simulation_pipeline():
    # 1. Configuration parameters
    n_samples = 5000
    seed_data = 42
    seed_sampling = 123
    true_p_a = 0.30
    true_p_b = 0.35
    n_simulations = 100000
    
    # 2. Generate simulated user telemetry data
    df = generate_user_logs(n_samples=n_samples, seed=seed_data, p_a=true_p_a, p_b=true_p_b)
    
    # 3. Aggregate data to compute trials and conversions
    metrics = aggregate_telemetry(df)
    
    # 4. Perform Bayesian posterior updates and draw Monte Carlo samples
    results = compute_posterior_sampling(
        metrics=metrics,
        n_simulations=n_simulations,
        seed=seed_sampling
    )
    
    # 5. Extract posterior samples and calculate statistics
    samples_a = results['samples_A']
    samples_b = results['samples_B']
    
    # Calculate point-by-point relative conversion lift for all posterior draws
    lift_samples = (samples_b - samples_a) / samples_a
    median_lift = float(np.median(lift_samples))
    
    # Calculate median conversion rates for each variant
    median_conv_a = float(np.median(samples_a))
    median_conv_b = float(np.median(samples_b))
    
    # 6. Output formatted summary metrics print block
    print("=" * 62)
    print("        BAYESIAN PRODUCT OPTIMIZATION SIMULATION PIPELINE        ")
    print("=" * 62)
    print("Configuration Parameters:")
    print(f"  - Simulated User Population:   {n_samples}")
    print(f"  - Target Conversion Rates:     Variant A = {true_p_a:.1%}, Variant B = {true_p_b:.1%}")
    print(f"  - Monte Carlo Draws (SciPy):   {n_simulations:,} simulations")
    print("-" * 62)
    print("Empirical Telemetry Performance:")
    for variant in ['A', 'B']:
        trials = metrics[variant]['trials']
        convs = metrics[variant]['conversions']
        emp_rate = convs / trials if trials > 0 else 0
        print(f"  - Variant {variant:1}: {trials:5} trials | {convs:4} conversions | Empirical Rate = {emp_rate:.2%}")
    print("-" * 62)
    print("Posterior Parameter Distributions:")
    print(f"  - Variant A Beta Posterior:    Alpha = {results['posteriors']['A']['alpha']:.1f}, Beta = {results['posteriors']['A']['beta']:.1f}")
    print(f"  - Variant B Beta Posterior:    Alpha = {results['posteriors']['B']['alpha']:.1f}, Beta = {results['posteriors']['B']['beta']:.1f}")
    print(f"  - Est. Median Conversion (A):  {median_conv_a:.2%} (95% CI: [{results['ci_A'][0]:.2%}, {results['ci_A'][1]:.2%}])")
    print(f"  - Est. Median Conversion (B):  {median_conv_b:.2%} (95% CI: [{results['ci_B'][0]:.2%}, {results['ci_B'][1]:.2%}])")
    print("-" * 62)
    print("Bayesian Decision Metrics:")
    print(f"  - P(Variant B > Variant A):    {results['prob_B_gt_A']:.4%}")
    print(f"  - Median Conv. Rate Lift:      {median_lift:+.2%} (95% CI: [{results['ci_lift'][0]:+.2%}, {results['ci_lift'][1]:+.2%}])")
    
    # Calculate VaR explanation
    var_val = results['lift_var_5pct']
    var_explanation = "Positive worst-case; zero downside risk vs A." if var_val > 0 else "Potential downside risk present."
    print(f"  - Value-at-Risk (VaR @ 5th%):   {var_val:+.2%} ({var_explanation})")
    print("=" * 62)

if __name__ == "__main__":
    run_simulation_pipeline()
