Temporal Causal Analysis (https://apxml.com/courses/causal-inference-ml-systems/chapter-5-causal-inference-temporal-dynamic/practice-temporal-causal-analysis)

A time-series causal discovery algorithm is applied to understand the dynamic relationships within a system based on observational time-series data. This simulates a common scenario in fields like economics, climate science, or system monitoring where multiple variables are recorded over time, and the underlying causal dependencies, including time lags, need to be inferred.

We will use the PCMCI algorithm, which you learned about earlier, as it's well-suited for discovering causal links in potentially high-dimensional time series while accounting for strong autocorrelation. We'll work with synthetic data for this exercise. Using synthetic data is advantageous here because we know the true underlying causal structure, allowing us to evaluate how well our discovery algorithm performs.
Setting Up the Environment and Data

First, ensure you have the necessary libraries installed. We'll primarily use tigramite for causal discovery and numpy for data generation.

```python
# Required libraries
# pip install tigramite numpy pandas matplotlib networkx
import numpy as np
import pandas as pd
import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr # Partial correlation test
from tigramite.plotting import plot_graph # For visualizing results
import matplotlib.pyplot as plt # Optional: for basic plotting
import networkx as nx # For graph manipulation if needed
```

Now, let's generate some synthetic time-series data based on a known Structural Causal Model (SCM) with time lags. We'll define a simple Vector Autoregressive (VAR) process with three variables (X,Y,ZX,Y,Z).

Let's define the ground truth causal relationships:

    XtXt​ is influenced by Xt−1Xt−1​ and Zt−1Zt−1​.
    YtYt​ is influenced by Yt−1Yt−1​ and Xt−1Xt−1​.
    ZtZt​ is influenced by Zt−1Zt−1​ and Yt−2Yt−2​.

The corresponding equations might look like this (with ϵϵ representing noise terms):
Xt=0.6Xt−1+0.4Zt−1+ϵX,t
Xt​=0.6Xt−1​+0.4Zt−1​+ϵX,t​
Yt=0.7Yt−1−0.5Xt−1+ϵY,t
Yt​=0.7Yt−1​−0.5Xt−1​+ϵY,t​
Zt=0.8Zt−1+0.3Yt−2+ϵZ,t
Zt​=0.8Zt−1​+0.3Yt−2​+ϵZ,t​

Let's generate data based on this structure:

```python
np.random.seed(42) # for reproducibility
T = 500 # Number of time points
N = 3   # Number of variables (X, Y, Z)

# Initialize data array
data = np.zeros((T, N))
# Noise terms
noise_std = 0.5
noise = np.random.normal(loc=0, scale=noise_std, size=(T, N))

# Generate data according to the VAR process
# Start from t=2 because we need lags up to t-2
for t in range(2, T):
    data[t, 0] = 0.6 * data[t-1, 0] + 0.4 * data[t-1, 2] + noise[t, 0] # X_t
    data[t, 1] = 0.7 * data[t-1, 1] - 0.5 * data[t-1, 0] + noise[t, 1] # Y_t
    data[t, 2] = 0.8 * data[t-1, 2] + 0.3 * data[t-2, 1] + noise[t, 2] # Z_t

# Add variable names
var_names = ['X', 'Y', 'Z']

# Create a pandas DataFrame for easier handling (optional but good practice)
df = pd.DataFrame(data, columns=var_names)
print("Generated Data Shape:", df.shape)
print(df.head())

# Optional: Basic plot to visualize the time series
# plt.figure(figsize=(12, 6))
# for i, name in enumerate(var_names):
#     plt.plot(df.index, df[name], label=name)
# plt.title('Generated Time Series Data')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
```

Preparing Data for Tigramite

tigramite expects data in a specific format. We need to wrap our numpy array using its Dataframe object.

```python
# Convert numpy array to Tigramite's Dataframe format
data_tig = pp.Dataframe(data, var_names=var_names)
```

Running the PCMCI Algorithm

Now we can instantiate and run the PCMCI algorithm. We need to specify:

    The data (dataframe).
    The conditional independence test (cond_ind_test). We'll use Partial Correlation (ParCorr), suitable for linear-Gaussian relationships like our VAR process.
    The significance level (pc_alpha). This threshold determines which conditional independencies are statistically significant. A common starting point is 0.05.
    The maximum lag (tau_max). This defines the maximum time delay we want to investigate for causal links. Based on our ground truth (Yt−2Yt−2​ affects ZtZt​), we need tau_max to be at least 2. Let's set it to 3 to be safe.

```python
# Initialize the conditional independence test
parcorr = ParCorr(significance='analytic')

# Initialize PCMCI
pcmci = PCMCI(
    dataframe=data_tig,
    cond_ind_test=parcorr,
    verbosity=1 # Print some output during execution
)

# Run PCMCI algorithm
# We set pc_alpha=0.01 for potentially stricter filtering
# max_lag_parents includes lag 0 (contemporaneous) but usually excluded for time series causality
results = pcmci.run_pcmci(tau_max=3, pc_alpha=0.01)

# The results contain several useful attributes:
# results['graph']: The discovered causal graph (numpy array)
# results['p_matrix']: P-values of conditional independence tests
# results['val_matrix']: Test statistic values (e.g., partial correlations)
```

The results['graph'] is a numpy array where graph[i, j, tau] represents the link from variable j at time t−τt−τ to variable i at time tt.
Visualizing and Interpreting the Discovered Graph

tigramite provides plotting utilities to visualize the discovered time-series graph.

```python
# Define the ground truth graph links for comparison
# Format: (from_node, to_node, lag)
# Note: Indices are 0 for X, 1 for Y, 2 for Z
true_links = {
    (0, 0, 1): {}, # X_t-1 -> X_t
    (2, 0, 1): {}, # Z_t-1 -> X_t
    (1, 1, 1): {}, # Y_t-1 -> Y_t
    (0, 1, 1): {}, # X_t-1 -> Y_t
    (2, 2, 1): {}, # Z_t-1 -> Z_t
    (1, 2, 2): {}  # Y_t-2 -> Z_t
}

# Plot the graph discovered by PCMCI
# We can also provide the true links to visualize TP, FP, FN edges
pq = plot_graph(
    val_matrix=results['val_matrix'], # Shows strength of links
    graph=results['graph'],           # Shows discovered links (based on pc_alpha)
    var_names=var_names,
    link_colorbar_label='cross-MCI partial corr.',
    node_colorbar_label='auto-MCI partial corr.',
    figsize=(8, 4),
    node_size=0.8,
    arrow_linewidth=1.5,
    # Pass the true graph for comparison visualization
    # true_graph_nodes=true_links # Note: Check tigramite documentation for exact format if needed
    # For manual graphviz generation, we extract links below
)
# plt.show() # Display the plot generated by tigramite's plotter

# For a Graphviz representation (more standard graph format)
# Extract links from results['graph']
discovered_links = []
for i in range(N): # Target variable index
    for j in range(N): # Source variable index
        for lag in range(1, results['graph'].shape[2]): # Iterate through lags > 0
            if results['graph'][i, j, lag] == '-->': # Check if a directed link exists
                source_node = f"{var_names[j]}(t-{lag})"
                target_node = f"{var_names[i]}(t)"
                discovered_links.append((source_node, target_node))

# Generate Graphviz DOT string
dot_string = "digraph TimeSeriesCausalGraph {\n"
dot_string += "  rankdir=LR;\n" # Left-to-right layout often better for time
dot_string += "  node [shape=ellipse, style=filled, fillcolor=\"#a5d8ff\"];\n" # Node style
dot_string += "  edge [color=\"#495057\"];\n" # Edge style
for source, target in discovered_links:
    dot_string += f"  \"{source}\" -> \"{target}\";\n"
dot_string += "}"

print("\nGraphviz DOT format:")
print(dot_string)
```

X(t-1) X(t) Y(t) Z(t-1) Z(t) Y(t-1) Y(t-2)

    Discovered causal graph from PCMCI applied to the synthetic VAR data. Nodes represent variables at specific time points (e.g., X(t−1)X(t−1) is variable X at time t−1t−1). Arrows indicate inferred causal relationships based on conditional independence tests.

Interpretation:

Compare the generated graph (dot_string output or the tigramite plot) with the ground truth we defined:

    Xt←Xt−1Xt​←Xt−1​ (Correctly found)
    Xt←Zt−1Xt​←Zt−1​ (Correctly found)
    Yt←Yt−1Yt​←Yt−1​ (Correctly found)
    Yt←Xt−1Yt​←Xt−1​ (Correctly found)
    Zt←Zt−1Zt​←Zt−1​ (Correctly found)
    Zt←Yt−2Zt​←Yt−2​ (Correctly found)

In this ideal scenario with clean, linear data matching the assumptions of the partial correlation test, PCMCI successfully recovered the true causal structure.
Evaluation and Analysis

"* Accuracy: In scenarios, you won't have the ground truth. Evaluating the quality of the discovered graph often relies on domain knowledge, consistency checks across different parameters (e.g., pc_alpha), or validation using interventional data if available. With synthetic data, we can calculate metrics like precision and recall for edge discovery."

    Parameter Sensitivity: The choice of pc_alpha and tau_max is significant.
        A lower pc_alpha (e.g., 0.01) makes the algorithm stricter, potentially leading to fewer false positives but more false negatives (missed links). A higher value (e.g., 0.05, 0.1) does the opposite. Sensitivity analysis by varying pc_alpha is recommended.
        tau_max should be chosen based on domain knowledge about the plausible time delays in the system. Setting it too low will miss longer-range dependencies, while setting it too high increases computational cost and the risk of finding spurious correlations.
    Assumptions: PCMCI relies on assumptions like causal sufficiency (no significant unobserved common causes), stationarity of the time series (or methods to handle non-stationarity), and the appropriateness of the chosen conditional independence test (ParCorr assumes linear relationships). If these are violated, the results might be inaccurate. For non-linear relationships, alternative tests like GPDC (Gaussian Process Distance Correlation) or CMIsymb (based on mutual information) within tigramite could be explored, though they require more data and computation.
    Contemporaneous Links: We focused on lagged effects (τ>0τ>0). Discovering contemporaneous links (Xt→YtXt​→Yt​) is more challenging due to symmetry issues and often requires additional assumptions or techniques (like those discussed for non-temporal DAGs or SVAR models). PCMCI primarily focuses on identifying lagged dependencies.

"This practical exercise demonstrated how to apply a sophisticated time-series causal discovery algorithm like PCMCI. You generated data, prepared it, ran the algorithm, and interpreted the resulting causal graph. Remember that applying these methods to data requires careful attention to the underlying assumptions, sensitivity to parameters, and validation against domain expertise."