# Bayesian Product Activation Optimization Engine

When a tech platform wants to test a new product design (like a redesigned signup flow), traditional testing frameworks often rely on frequentist statistical models that require large, rigid sample sizes and yield abstract metrics that can be challenging to translate into direct business impact. This project solves that by using Bayesian statistics, an intuitive, decision-oriented approach that models A/B testing as a process of continuous learning from experience. 

Instead of relying solely on abstract frequentist metrics (like p-values), this engine translates posterior probability distributions into clear, actionable business answers: *"There is a 100% probability that our new onboarding flow (Variant B) outperforms the current design (Control A), yielding a +22.7% lift with zero downside risk."*

## Project Framework Architecture
* **The Challenge:** Traditional A/B testing frameworks often require teams to collect fixed, massive sample sizes before declaring a result, yielding abstract statistics. This Bayesian framework addresses these limitations by mapping out the entire range of likely outcomes (Credible Intervals) and calculating the exact *Value-at-Risk (VaR)* to quantify potential downside before deploying.
* **The Solution:** A modular Python-based simulation pipeline. It takes raw user activity data from two onboarding designs, treats their signup rates as active probability curves, and updates our statistical "beliefs" as new data rolls in.
* **How it's done:** The codebase automatically:
  1. Aggregates granular telemetry logs.
  2. Computes exact conjugate updates (updating Beta-Binomial probability distributions).
  3. Runs $100,000$ Monte Carlo simulations to model every possible scenario.
  4. Generates a beautifully formatted console report showing estimated conversion rates, high-probability ranges, and risk thresholds.
  
## Business Challenge
A mobile healthcare platform needs to maximize user onboarding activation rates within the first 7 days of subscription ingress. The control application setup interface (Control A) uses a comprehensive, multi-step clinical form, while the experimental alternative setup (Variant B) streamlines form layout patterns and groups text entries into cleaner sections. 

The analytical objective is to determine if Variant B reduces conversion barriers and to quantify the strategic implications of deploying it across the entire platform.

## Analytical Approach & Methodology
Traditional frequentist approaches evaluate the binary probability of observing data assuming a null hypothesis ($H_0: \theta_B = \theta_A$). This framework instead isolates continuous variations over parameter spaces using Bayesian updates:

1. **Prior Selection:** Assumes a standard uninformative flat prior ($\text{Beta}(\alpha=1, \beta=1)$) to represent total ignorance before collecting telemetry records.
2. **Conjugate Parameter Calculations:** Updates prior matrices sequentially using observed trial events to form updated posterior structures:
   $$\alpha_{\text{posterior}} = \alpha_{\text{prior}} + \text{successes}$$
   $$\beta_{\text{posterior}} = \beta_{\text{prior}} + \text{failures}$$
3. **Simulation Computation:** Runs 10,000 independent Random Variate draws from the updated models to map distribution curves, outputting precise interval estimates and continuous value-at-risk thresholds.

## Production Setup & Deployment Validation

To run this pipeline locally and output the metrics report directly into your console configuration array, execute:

```bash
# Clone the analytical module
git clone https://github.com/vyvuong159/bayesian-product-optimization.git
cd bayesian-product-optimization

# Standardize software dependencies via pip environment tools
pip install -r requirements.txt

# Run the complete data simulation pipeline workflow
python run_pipeline.py
```

## Sample Simulation Output Report
Executing the pipeline will output the following report in your terminal:

```text
==============================================================
        BAYESIAN PRODUCT OPTIMIZATION SIMULATION PIPELINE        
==============================================================
Configuration Parameters:
  - Simulated User Population:   5000
  - Target Conversion Rates:     Variant A = 30.0%, Variant B = 35.0%
  - Monte Carlo Draws (SciPy):   100,000 simulations
--------------------------------------------------------------
Empirical Telemetry Performance:
  - Variant A:  2545 trials |  738 conversions | Empirical Rate = 29.00%
  - Variant B:  2455 trials |  874 conversions | Empirical Rate = 35.60%
--------------------------------------------------------------
Posterior Parameter Distributions:
  - Variant A Beta Posterior:    Alpha = 739.0, Beta = 1808.0
  - Variant B Beta Posterior:    Alpha = 875.0, Beta = 1582.0
  - Est. Median Conversion (A):  29.01% (95% CI: [27.26%, 30.81%])
  - Est. Median Conversion (B):  35.61% (95% CI: [33.73%, 37.52%])
--------------------------------------------------------------
Bayesian Decision Metrics:
  - P(Variant B > Variant A):    100.0000%
  - Median Conv. Rate Lift:      +22.74% (95% CI: [+13.22%, +33.05%])
  - Value-at-Risk (VaR @ 5th%):   +14.71% (Positive worst-case; zero downside risk vs A.)
==============================================================
```