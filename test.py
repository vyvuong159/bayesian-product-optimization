from src.data_generation import generate_user_logs
from src.bayesian_engine import aggregate_telemetry, compute_posterior_sampling

def test_end_to_end_pipeline():
    print("=== Testing End-to-End Bayesian Optimization Pipeline ===")
    
    # 1. Generate telemetry logs
    n_samples = 5000
    seed = 42
    p_a = 0.30
    p_b = 0.35
    print(f"Generating {n_samples} user logs (Seed: {seed}, True P(A): {p_a}, True P(B): {p_b})...")
    df = generate_user_logs(n_samples=n_samples, seed=seed, p_a=p_a, p_b=p_b)
    
    # 2. Aggregate telemetry data
    print("\nAggregating telemetry data...")
    metrics = aggregate_telemetry(df)
    print("Aggregated Metrics:")
    for variant, data in metrics.items():
        print(f"  Variant {variant}: {data['conversions']} conversions / {data['trials']} trials (Empirical rate: {data['conversions']/data['trials']:.4f})")
        
    # 3. Compute posterior updates and sampling
    n_simulations = 100000
    print(f"\nComputing posteriors and running {n_simulations} simulations...")
    results = compute_posterior_sampling(metrics, n_simulations=n_simulations, seed=123)
    
    print("\nPosterior Parameters:")
    for variant, post in results['posteriors'].items():
        print(f"  Variant {variant}: Alpha={post['alpha']}, Beta={post['beta']}")
        
    print("\nBayesian Estimation Results:")
    print(f"  Probability that Variant B is superior to A (P(B > A)): {results['prob_B_gt_A']:.4%}")
    print(f"  Expected relative lift of B over A: {results['expected_lift']:.2%}")

if __name__ == "__main__":
    test_end_to_end_pipeline()
