DML and Causal Forest Implementation (https://apxml.com/courses/causal-inference-ml-systems/chapter-3-high-dimensional-effect-estimation/practice-dml-causal-forests)

Double Machine Learning (DML) and Causal Forests are techniques for estimating effects in high-dimensional settings. These estimators are implemented using common Python libraries, allowing their application to realistic datasets. Simulated data, where the ground truth is known, will be used to verify the performance of these implementations.

Our goal is to estimate both the Average Treatment Effect (ATE) using DML and the Conditional Average Treatment Effects (CATE) using Causal Forests.
Setup and Data Simulation

First, ensure you have the necessary libraries installed. We'll primarily use econml, scikit-learn, pandas, and numpy.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV
from econml.dml import LinearDML, CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter
import matplotlib.pyplot as plt # Used only for plotting example below
```

Let's simulate data with high-dimensional confounders (XX), a binary treatment (TT), and an outcome (YY). We'll design the simulation such that the treatment effect is heterogeneous, depending on one of the confounders.

```python
# Simulation parameters
n_samples = 5000  # Number of samples
n_features = 20   # Number of confounders
true_ate = 1.5    # Base average treatment effect
heterogeneity_slope = 0.5 # Slope for heterogeneity based on X0

# Generate confounders
np.random.seed(42)
X = np.random.normal(0, 1, size=(n_samples, n_features))
X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])

# Generate treatment assignment (propensity score depends on X)
# Simple logistic model for propensity
propensity_coeffs = np.random.uniform(-0.5, 0.5, size=n_features)
propensity_logit = X @ propensity_coeffs + np.random.normal(0, 0.1, size=n_samples)
propensity = 1 / (1 + np.exp(-propensity_logit))
T = np.random.binomial(1, propensity, size=n_samples)

# Generate outcome (depends on X, T, and incorporates heterogeneity)
# Y = linear effect of X + treatment effect * T + noise
outcome_coeffs = np.random.uniform(0, 1, size=n_features)
# True CATE = true_ate + heterogeneity_slope * X[:, 0]
true_cate = true_ate + heterogeneity_slope * X[:, 0]
Y = X @ outcome_coeffs + true_cate * T + np.random.normal(0, 0.5, size=n_samples)

print(f"Simulated data shapes:")
print(f"X: {X.shape}, T: {T.shape}, Y: {Y.shape}")
print(f"True Average Treatment Effect (based on simulation): {np.mean(true_cate):.4f}")
```
                
This setup mimics a common scenario where many features potentially confound the treatment-outcome relationship, and the treatment's effectiveness varies across individuals based on their characteristics (here, specifically X0X0​).
Implementing Double Machine Learning (DML) for ATE

DML estimates the ATE by using machine learning models to partial out the effects of confounders XX from both the treatment TT and the outcome YY. This involves fitting two nuisance models:

    Outcome model: E[Y∣X]E[Y∣X]
    Treatment model: E[T∣X]E[T∣X] (propensity score model if T is binary)

We then estimate the effect using the residuals from these models. The econml library simplifies this process. We'll use LinearDML which assumes a linear final stage for estimating the constant ATE.

```python
# Define nuisance models
# For outcome model E[Y|X]
model_y = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
# For treatment model E[T|X] (propensity)
model_t = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

# Instantiate LinearDML estimator
# We use discrete_treatment=True since T is binary
dml_estimator = LinearDML(model_y=model_y,
                          model_t=model_t,
                          discrete_treatment=True,
                          random_state=123)

# Fit the estimator
# We provide outcome Y, treatment T, confounders X, and optionally W (no effect modifiers here)
dml_estimator.fit(Y, T, X=X)

# Get the ATE estimate and confidence interval
ate_estimate = dml_estimator.effect(T=1) # Effect of going from T=0 to T=1
ate_ci = dml_estimator.effect_interval(T=1, alpha=0.05) # 95% CI

print(f"DML Estimated ATE: {ate_estimate[0]:.4f}")
print(f"95% Confidence Interval: [{ate_ci[0]:.4f}, {ate_ci[1]:.4f}]")
# Compare with the true ATE calculated earlier
print(f"True Average Treatment Effect: {np.mean(true_cate):.4f}")
```

The LinearDML handles the cross-fitting procedure internally to prevent overfitting and provides standard errors for inference. Compare the estimated ATE and its confidence interval to the true average effect from our simulation. They should be reasonably close, demonstrating DML's ability to recover the average effect despite high-dimensional confounding.
Implementing Causal Forests for CATE

While DML provides an estimate of the average effect, Causal Forests aim to reveal heterogeneity in treatment effects. They adapt the random forest algorithm to estimate CATE, E[Y(1)−Y(0)∣X=x]E[Y(1)−Y(0)∣X=x]. econml provides CausalForestDML, which integrates the DML residualization approach within the forest structure.

```python
# Instantiate CausalForestDML estimator
# It uses DML principles for orthogonalization within the forest splits
# We can specify nuisance models or use defaults (often Gradient Boosting)
cf_estimator = CausalForestDML(model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
                               model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
                               discrete_treatment=True,
                               n_estimators=1000, # More trees are generally better for forests
                               min_samples_leaf=10,
                               max_depth=10,
                               random_state=123)

# Fit the Causal Forest
cf_estimator.fit(Y, T, X=X)

# Estimate CATE for all samples in the dataset
cate_estimates = cf_estimator.effect(X=X)

print(f"Shape of CATE estimates: {cate_estimates.shape}")
print(f"Example CATE estimates (first 5): {cate_estimates[:5].round(4)}")
```
                
The cf_estimator.effect(X=X) returns an array of CATE estimates, one for each sample based on its features XX.
Visualizing CATE Heterogeneity

To understand how the treatment effect varies, we can visualize the estimated CATE against the feature driving the heterogeneity (X0X0​ in our simulation).
```python
# Create a scatter plot of estimated CATE vs X0
# For visualization, let's sample points to avoid overplotting
sample_indices = np.random.choice(n_samples, 500, replace=False)
x0_sample = X[sample_indices, 0]
cate_sample = cate_estimates[sample_indices]

# Define the Plotly chart data and layout
plotly_fig = {
    "data": [
        {
            "type": "scatter",
            "mode": "markers",
            "x": x0_sample.tolist(), # Use sampled data
            "y": cate_sample.tolist(), # Use sampled data
            "marker": {
                "color": "#228be6", # blue
                "size": 6,
                "opacity": 0.6
            },
            "name": "Estimated CATE"
        },
        # Add line showing the true relationship for reference
        {
            "type": "scatter",
            "mode": "lines",
            "x": sorted(x0_sample),
            "y": (true_ate + heterogeneity_slope * np.sort(x0_sample)).tolist(),
            "line": {
                "color": "#f03e3e", # red
                "width": 2,
                "dash": "dash"
            },
            "name": "True CATE"
        }
    ],
    "layout": {
        "title": "Estimated CATE vs. Feature X0",
        "xaxis": {"title": "Feature X0"},
        "yaxis": {"title": "Estimated CATE"},
        "showlegend": True,
        "legend": {"x": 0.01, "y": 0.99},
        "width": 700,
        "height": 450,
        "template": "plotly_white"
    }
}
```
Estimated Conditional Average Treatment Effects (CATE) plotted against the values of feature X0 for a sample of the data. The dashed red line indicates the true CATE relationship defined in the simulation (CATE=1.5+0.5∗X0CATE=1.5+0.5∗X0​).

The plot should show that the Causal Forest's CATE estimates generally follow the true positive slope, indicating that the model successfully captured the increasing treatment effect with higher values of X0X0​. The scatter represents the individual CATE estimates, which will naturally have some variance around the true line.

You can further interpret the CATE estimates using tools like econml.cate_interpreter.SingleTreeCateInterpreter to understand which features are most important for driving heterogeneity.
```python
# Interpret the CATE model with a simple tree
intrp = SingleTreeCateInterpreter(include_model_uncertainty=False, max_depth=2)
intrp.interpret(cf_estimator, X)

# Plot the interpretation tree (requires graphviz)
# intrp.plot(feature_names=X_df.columns, fontsize=12)
# The plot would show splits primarily on X0 if the model works well.
print("CATE Interpretation Tree Structure:")
print(intrp.text_summary(feature_names=X_df.columns))
```

This summary provides a simplified tree structure approximating the Causal Forest's CATE function, often highlighting the most influential features (like X0X0​ in our case).
Summary and Next Steps

This practice session demonstrated the implementation of Double Machine Learning for ATE estimation and Causal Forests for CATE estimation in a high-dimensional setting using econml. We saw how DML effectively recovers the average effect by handling confounders with nuisance models, and how Causal Forests can reveal heterogeneity in treatment effects.

Important takeaways:

    DML relies on accurate modeling of nuisance functions (E[Y∣X]E[Y∣X] and E[T∣X]E[T∣X]). The choice of ML models for these tasks is important.
    Causal Forests build upon DML principles to estimate how effects vary with individual characteristics XX.
    Using simulated data allows for verification of the methods by comparing estimates against known ground truth.

From here, you can experiment with different models within DML and Causal Forests (e.g., LassoCV vs. GradientBoosting), tune hyperparameters, and apply these techniques to your own datasets. Remember to carefully examine the assumptions underlying these methods, particularly the unconfoundedness assumption (conditional ignorability) given the observed covariates XX. The next chapter explores situations where this assumption is violated due to unobserved confounding.