IV and RDD Analysis (https://apxml.com/courses/causal-inference-ml-systems/chapter-4-addressing-hidden-bias/practice-iv-rdd-analysis)

Instrumental Variables (IV) and Regression Discontinuity Designs (RDD) are powerful techniques for tackling unobserved confounding. Practical, hands-on examples using Python demonstrate how to implement these methods to estimate causal effects in scenarios where simple regression would yield biased results. A working Python environment with libraries like pandas, numpy, statsmodels, linearmodels, and plotly is assumed.
Setting the Stage: The Problem of Unobserved Confounding

Recall the core challenge: we want to estimate the causal effect of a treatment TT on an outcome YY, but there exists an unobserved variable UU that affects both TT and YY. A naive regression of YY on TT (and observed covariates XX) will produce a biased estimate of the causal effect because it fails to account for UU.
Y=α+βcausalT+γ′X+δU+ϵT=π0+π1Z+π2′X+π3U+ν
Y=α+βcausal​T+γ′X+δU+ϵT=π0​+π1​Z+π2′​X+π3​U+ν

If δ≠0δ=0 and π3≠0π3​=0, then TT is endogenous, and standard regression estimates for βcausalβcausal​ are inconsistent. IV and RDD offer ways to obtain consistent estimates under specific structural assumptions.
Implementing Instrumental Variables (IV) Estimation

IV methods leverage an instrument ZZ, a variable that satisfies three conditions:

    Relevance: ZZ must be correlated with the treatment TT (i.e., π1≠0π1​=0 in the equation above, after conditioning on XX).
    Exclusion: ZZ affects the outcome YY only through its effect on TT. It has no direct path to YY (i.e., ZZ does not appear in the equation for YY, once TT and XX are included).
    Independence (Ignorability): ZZ is independent of the unobserved confounder UU (or more formally, independent of the potential outcomes given covariates).

Finding a valid instrument often requires significant domain knowledge and careful justification.
Scenario: Estimating Returns to Education

Let's examine a classic econometrics problem: estimating the causal effect of years of schooling (TT) on log-wages (YY). 'Innate ability' (UU) is a likely unobserved confounder, affecting both educational attainment and wages directly. A potential instrument ZZ could be proximity to a college when growing up, or as in Angrist and Krueger's (1991) famous study, the quarter of birth, which affects years of schooling due to compulsory schooling laws but is arguably unrelated to innate ability or future wages directly.
Two-Stage Least Squares (2SLS)

The most common IV estimator is Two-Stage Least Squares (2SLS).

    First Stage: Regress the endogenous treatment TT on the instrument ZZ and any observed exogenous covariates XX. Obtain the predicted values T^T^.
    T=π^0+π^1Z+π^2′X+residual
    T=π^0​+π^1​Z+π^2′​X+residual
    Second Stage: Regress the outcome YY on the predicted treatment T^T^ from the first stage and the observed covariates XX.
    Y=α^+β^2SLST^+γ^′X+residual
    Y=α^+β^​2SLS​T^+γ^​′X+residual

The coefficient β^2SLSβ^​2SLS​ is the consistent IV estimate of the causal effect.
Implementation with linearmodels

The linearmodels library provides a convenient interface for IV estimation. Let's simulate some data where OLS fails but IV works. Assume ability (UU) is unobserved. qob (quarter of birth) is our instrument ZZ. schooling is TT, log_wage is YY, and experience is an observed covariate XX.

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# Simulate data (Illustrative - real data analysis needed for actual insights)
np.random.seed(42)
n_samples = 1000
experience = np.random.uniform(5, 25, n_samples)
ability = np.random.normal(0, 1, n_samples) # Unobserved
qob = np.random.randint(1, 5, n_samples) # Instrument (e.g., quarter of birth)

# Schooling affected by ability and instrument
schooling = 10 + 0.5 * qob + 1.0 * ability + 0.1 * experience + np.random.normal(0, 1, n_samples)
# Log-wage affected by schooling, ability, and experience
log_wage = 5 + 0.1 * schooling + 0.5 * ability + 0.05 * experience + np.random.normal(0, 0.5, n_samples)

df = pd.DataFrame({
    'log_wage': log_wage,
    'schooling': schooling,
    'experience': experience,
    'qob': qob,
    'ability': ability # Keep for comparison, but treat as unobserved in estimation
})

# Add a constant for regression models
df = sm.add_constant(df, prepend=False)

# --- Naive OLS (Biased due to omitting ability) ---
ols_model = sm.OLS(df['log_wage'], df[['schooling', 'experience', 'const']])
ols_results = ols_model.fit()
print("--- OLS Results (Potentially Biased) ---")
print(ols_results.summary(yname='log_wage', xname=['schooling', 'experience', 'const']))
print(f"\nOLS estimate for schooling effect: {ols_results.params['schooling']:.4f}\n")

# --- IV (2SLS) Estimation ---
# Dependent: log_wage
# Exogenous regressors: experience, const
# Endogenous regressor: schooling
# Instrument: qob

iv_model = IV2SLS(dependent=df['log_wage'],
                  exog=df[['experience', 'const']],
                  endog=df['schooling'],
                  instruments=df['qob'])
iv_results = iv_model.fit(cov_type='standard') # Use standard standard errors

print("\n--- IV (2SLS) Results ---")
print(iv_results.summary)

# Check First Stage F-statistic for weak instrument
print("\n--- First Stage Diagnostics ---")
print(iv_results.first_stage)

# The true causal effect is 0.1
# The OLS estimate is biased upwards because ability positively affects both schooling and wages.
# The IV estimate should be closer to 0.1, provided the instrument is valid and strong enough.
```

Interpreting the Output:

Compare the coefficient for schooling from the OLS results with the IV results. The OLS estimate is likely biased (in our simulation, upwardly biased because higher ability leads to more schooling and higher wages). The IV estimate, assuming the instrument is valid, should be closer to the true causal effect (0.1 in our simulation).

Pay close attention to the First Stage Diagnostics. The F-statistic tests the relevance of the instrument(s). A common rule of thumb suggests an F-statistic greater than 10 indicates the instrument is sufficiently strong, although context matters. Weak instruments (low F-statistic) lead to unreliable and biased IV estimates.
Implementing Regression Discontinuity Designs (RDD)

RDD applies when assignment to a treatment TT is determined, either sharply or fuzzily, by whether an observed running variable RR crosses a known threshold cc.

    Sharp RDD: Treatment is perfectly determined by the threshold: T=1T=1 if R≥cR≥c, and T=0T=0 if R<cR<c.
    Fuzzy RDD: The probability of receiving treatment changes discontinuously at the threshold: lim⁡r→c+P(T=1∣R=r)≠lim⁡r→c−P(T=1∣R=r)limr→c+​P(T=1∣R=r)=limr→c−​P(T=1∣R=r).

The core idea is that units just below the cutoff (R=c−ϵR=c−ϵ) and just above the cutoff (R=c+ϵR=c+ϵ) are likely similar in terms of both observed and unobserved characteristics. Therefore, any discontinuous jump in the average outcome YY at the threshold cc can be attributed to the causal effect of the treatment. RDD estimates a Local Average Treatment Effect (LATE) specifically for units near the cutoff.
Scenario: Scholarship Effect on Graduation Rate

Imagine a university grants a scholarship (T=1T=1) to students scoring at or above a threshold c=700c=700 on an entrance exam (RR), and no scholarship (T=0T=0) for scores below 700. We want to estimate the effect of the scholarship on the probability of graduation (YY).
Visualizing the Discontinuity

An important first step in RDD is visualization. Plot the outcome YY against the running variable RR. Look for a jump or discontinuity at the cutoff cc. Often, people plot averages within bins of the running variable.

```python
import plotly.graph_objects as go

# Simulate RDD data
np.random.seed(123)
n_samples = 2000
cutoff = 700
# Running variable (exam score) centered around cutoff
running_var = np.random.normal(cutoff, 50, n_samples)
# Treatment assignment (Sharp RDD)
treatment = (running_var >= cutoff).astype(int)
# Potential outcomes (what graduation prob would be with/without scholarship)
# Assume scholarship increases graduation prob by 0.15
baseline_prob = 0.5 + 0.0005 * (running_var - cutoff) # Baseline trend
potential_outcome_0 = baseline_prob + np.random.normal(0, 0.1, n_samples)
potential_outcome_1 = potential_outcome_0 + 0.15
# Observed outcome
observed_outcome = potential_outcome_0 * (1 - treatment) + potential_outcome_1 * treatment
# Clip outcome to be between 0 and 1 (probability)
observed_outcome = np.clip(observed_outcome, 0, 1)

rdd_df = pd.DataFrame({
    'score': running_var,
    'scholarship': treatment,
    'graduated': observed_outcome
})

# Bin data for visualization
bins = np.linspace(rdd_df['score'].min(), rdd_df['score'].max(), 50)
rdd_df['score_bin'] = pd.cut(rdd_df['score'], bins, right=False)
binned_data = rdd_df.groupby('score_bin', observed=True).agg(
    mean_score=('score', 'mean'),
    mean_graduated=('graduated', 'mean'),
    n_obs=('score', 'size')
).reset_index()
binned_data = binned_data.dropna(subset=['mean_score']) # Drop empty bins

# Create Plotly chart
fig = go.Figure()

# Scatter plot of binned averages
fig.add_trace(go.Scatter(
    x=binned_data['mean_score'],
    y=binned_data['mean_graduated'],
    mode='markers',
    marker=dict(
        color=np.where(binned_data['mean_score'] < cutoff, '#4263eb', '#f76707'), # Blue below, Orange above
        size=np.sqrt(binned_data['n_obs']) * 1.5, # Size by number of observations
        line_width=1,
        line_color='#495057'
    ),
    name='Binned Averages'
))

# Add vertical line for cutoff
fig.add_vline(x=cutoff, line_width=2, line_dash="dash", line_color="#868e96", name=f'Cutoff = {cutoff}')

# Add local regression lines (Fit separately for illustration - real analysis uses specific RDD methods)
# Note: Simple linear fits shown here; local polynomial is standard.
score_below = binned_data[binned_data['mean_score'] < cutoff]['mean_score']
grad_below = binned_data[binned_data['mean_score'] < cutoff]['mean_graduated']
params_below = np.polyfit(score_below, grad_below, 1)
x_below = np.linspace(score_below.min(), cutoff, 50)
y_below = np.polyval(params_below, x_below)

score_above = binned_data[binned_data['mean_score'] >= cutoff]['mean_score']
grad_above = binned_data[binned_data['mean_score'] >= cutoff]['mean_graduated']
params_above = np.polyfit(score_above, grad_above, 1)
x_above = np.linspace(cutoff, score_above.max(), 50)
y_above = np.polyval(params_above, x_above)

fig.add_trace(go.Scatter(x=x_below, y=y_below, mode='lines', line=dict(color='#1c7ed6', width=2), name='Fit (Score < 700)'))
fig.add_trace(go.Scatter(x=x_above, y=y_above, mode='lines', line=dict(color='#f76707', width=2), name='Fit (Score >= 700)'))

fig.update_layout(
    title_text='RDD Visualization: Graduation Rate vs. Entrance Exam Score',
    xaxis_title='Entrance Exam Score (Running Variable)',
    yaxis_title='Average Graduation Rate',
    legend_title='Legend',
    plot_bgcolor='#e9ecef',
    paper_bgcolor='white',
    font=dict(family="Arial, sans-serif", size=12, color="#495057")
)

# Show plot JSON (replace with actual rendering in web context)
# fig.show()
# Example fixed JSON for embedding:
```plotly
{"data": [{"type": "scatter", "mode": "markers", "x": [620, 635, 650, 665, 680, 695, 705, 720, 735, 750, 765, 780], "y": [0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80], "marker": {"color": ["#4263eb", "#4263eb", "#4263eb", "#4263eb", "#4263eb", "#4263eb", "#f76707", "#f76707", "#f76707", "#f76707", "#f76707", "#f76707"], "size": [8, 9, 10, 11, 10, 9, 8, 9, 10, 11, 10, 9], "line": {"width": 1, "color": "#495057"}}, "name": "Binned Averages"}, {"type": "scatter", "x": [620, 699.9], "y": [0.54, 0.595], "mode": "lines", "line": {"color": "#1c7ed6", "width": 2}, "name": "Fit (Score < 700)"}, {"type": "scatter", "x": [700, 780], "y": [0.745, 0.80], "mode": "lines", "line": {"color": "#f76707", "width": 2}, "name": "Fit (Score >= 700)"}], "layout": {"xaxis": {"range": [610, 790], "title": "Entrance Exam Score (Running Variable)"}, "yaxis": {"range": [0.45, 0.85], "title": "Average Graduation Rate"}, "shapes": [{"type": "line", "x0": 700, "y0": 0.45, "x1": 700, "y1": 0.85, "line": {"color": "#868e96", "width": 2, "dash": "dash"}}], "title": {"text": "RDD: Graduation Rate vs. Entrance Exam Score"}, "showlegend": true, "legend": {"title": {"text": "Legend"}}, "plot_bgcolor": "#e9ecef", "paper_bgcolor": "white", "font": {"family": "Arial, sans-serif", "size": 12, "color": "#495057"}}}
```
The plot shows average graduation rates within score bins. A clear jump is visible around the cutoff score of 700, suggesting a positive effect of the scholarship. Fitted lines help visualize the discontinuity.

Estimating the LATE

The standard approach is to use local polynomial regression. We fit separate polynomials to the data within a chosen bandwidth hh on either side of the cutoff cc. The difference in the intercepts of these regressions at the cutoff cc is the RDD estimate of the LATE.
β^RDD=lim⁡r→c+E[Y∣R=r]−lim⁡r→c−E[Y∣R=r]
β^​RDD​=r→c+lim​E[Y∣R=r]−r→c−lim​E[Y∣R=r]

While you can implement this manually using regression packages, specialized libraries like rdrobust (available in R and Python) automate bandwidth selection (e.g., using Imbens-Kalyanaraman optimal bandwidth) and estimation with standard errors.

```python
# Simplified implementation using statsmodels OLS for illustration
# Proper RDD requires local polynomials and optimal bandwidth selection
# Consider using rdrobust package for serious analysis

bandwidth = 30 # Example bandwidth - needs careful selection!
df_rdd_local = rdd_df[(rdd_df['score'] >= cutoff - bandwidth) & (rdd_df['score'] < cutoff + bandwidth)].copy()

# Create interaction term for local linear regression
df_rdd_local['score_centered'] = df_rdd_local['score'] - cutoff
df_rdd_local['treat_x_score_centered'] = df_rdd_local['scholarship'] * df_rdd_local['score_centered']

# Add constant
df_rdd_local = sm.add_constant(df_rdd_local, prepend=False)

# Regression: outcome ~ treat + score_centered + treat:score_centered
rdd_model = sm.OLS(df_rdd_local['graduated'],
                   df_rdd_local[['scholarship', 'score_centered', 'treat_x_score_centered', 'const']])
rdd_results = rdd_model.fit(cov_type='robust') # Use robust SEs

print("\n--- RDD Estimation (Local Linear Regression - Illustrative) ---")
print(rdd_results.summary(yname='graduated', xname=['scholarship (T)', 'score_centered (R-c)', 'T*(R-c)', 'const']))

# The coefficient on 'scholarship' is the RDD estimate of the LATE at the cutoff
rdd_estimate = rdd_results.params['scholarship']
print(f"\nRDD estimate (LATE) of scholarship effect: {rdd_estimate:.4f}")

# Compare to the true effect of 0.15 in simulation
```

Interpretation and Key Points:

    The coefficient on the treatment dummy (scholarship in the example) represents the estimated jump in the outcome at the cutoff, which is the LATE.
    Bandwidth Selection: The choice of bandwidth hh is critical. Too narrow, and the estimate is noisy. Too wide, and the assumption of similarity across the threshold may be violated (bias-variance tradeoff). Use data-driven methods (like those in rdrobust).
    Functional Form: Usually, local linear (polynomial order 1) or local quadratic (order 2) regression is preferred over higher-order polynomials.
    Fuzzy RDD: If the discontinuity is in the probability of treatment (not deterministic), Fuzzy RDD is needed. This is typically implemented using 2SLS, where the dummy variable indicating assignment above the threshold (Z=1[R≥c]Z=1[R≥c]) serves as an instrument for the actual treatment status TT.
    Validity Checks: Perform checks like ensuring covariates don't jump discontinuously at the cutoff (McCrary density test) and running placebo tests using different cutoffs.

Conclusion

This practical section demonstrated how to implement IV (using 2SLS) and RDD (using local regression) in Python. These methods are invaluable additions to your toolkit when facing potential unobserved confounding, allowing for causal effect estimation under specific, albeit strong, assumptions. Always critically evaluate the validity of IV (relevance, exclusion, independence) and RDD (continuity of potential outcomes and covariates, no manipulation of the running variable) assumptions in your specific application context using domain knowledge and diagnostic tests. Mastering these techniques enables more credible causal conclusions from observational data, a significant step towards building more reliable ML systems.