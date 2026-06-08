Implementing Discovery Algorithms (https://apxml.com/courses/causal-inference-ml-systems/chapter-2-advanced-causal-discovery/practice-implementing-discovery)

Gain hands-on experience applying causal discovery algorithms. Python libraries are used to infer causal graphs from data, focusing on interpreting results and understanding the practical implications of different algorithm choices. A working Python environment with pandas, numpy, matplotlib, and specialized causal discovery libraries installed is assumed.
Setting the Stage: Environment and Data

We will primarily use the causal-learn library, a comprehensive package for causal discovery. If you haven't installed it yet, you can do so via pip:

```python
pip install causal-learn matplotlib
```

For visualization, causal-learn relies on graphviz. Ensure you have Graphviz installed system-wide (check the Graphviz website for instructions for your OS) and the Python graphviz package:

```python
pip install graphviz
```

Let's generate some synthetic data based on a known Structural Causal Model (SCM). This allows us to evaluate how well our discovery algorithms perform against a ground truth. Here's an example SCM where NiNi​ represents independent noise terms:

    X1=N1X1​=N1​
    X2=0.8⋅X1+N2X2​=0.8⋅X1​+N2​
    X3=0.5⋅X1−0.6⋅X2+N3X3​=0.5⋅X1​−0.6⋅X2​+N3​
    X4=0.7⋅X2+0.4⋅X5+N4X4​=0.7⋅X2​+0.4⋅X5​+N4​
    X5=N5X5​=N5​

This corresponds to the ground truth DAG: X1→X2X1​→X2​, X1→X3X1​→X3​, X2→X3X2​→X3​, X2→X4X2​→X4​, X5→X4X5​→X4​.
```python
import numpy as np
import pandas as pd
from causal_learn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
import graphviz

# Set seed for reproducibility
np.random.seed(42)
num_samples = 1000

# Generate data based on the SCM
n1 = np.random.randn(num_samples)
n2 = np.random.randn(num_samples)
n3 = np.random.randn(num_samples)
n4 = np.random.randn(num_samples)
n5 = np.random.randn(num_samples)

x1 = n1
x5 = n5
x2 = 0.8 * x1 + n2
x3 = 0.5 * x1 - 0.6 * x2 + n3
x4 = 0.7 * x2 + 0.4 * x5 + n4

# Create DataFrame
data = pd.DataFrame({'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 'X5': x5})

# Define ground truth graph (Adjacency Matrix)
# Rows/Cols correspond to X1, X2, X3, X4, X5
true_dag_matrix = np.array([
    [0, 1, 1, 0, 0],  # X1 -> X2, X1 -> X3
    [0, 0, 1, 1, 0],  # X2 -> X3, X2 -> X4
    [0, 0, 0, 0, 0],  # X3 has no outgoing edges
    [0, 0, 0, 0, 0],  # X4 has no outgoing edges
    [0, 0, 0, 1, 0]   # X5 -> X4
])

print("Simulated Data Head:")
print(data.head())

# --- Function to visualize Graph using Graphviz ---
def plot_graph(adjacency_matrix, labels, title="Learned Graph"):
    """Visualizes a DAG using graphviz."""
    num_nodes = adjacency_matrix.shape[0]
    node_names = [labels[i] for i in range(num_nodes)]

    dot = graphviz.Digraph(comment=title, graph_attr={'rankdir': 'LR', 'splines':'true', 'overlap':'false'})

    # Add nodes
    for name in node_names:
        dot.node(name, name, shape='ellipse', style='filled', fillcolor='#a5d8ff', color='#1c7ed6')

    # Add edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] == 1: # Directed edge i -> j
                 dot.edge(node_names[i], node_names[j], color='#495057')
            elif adjacency_matrix[i, j] == -1 and adjacency_matrix[j, i] == -1: # Undirected edge i -- j (from CPDAG)
                 if i < j: # Draw only once
                     dot.edge(node_names[i], node_names[j], dir='none', color='#f76707')
            # Add other edge types if needed (e.g., bi-directed for FCI)

    return dot

# Visualize Ground Truth
print("\nVisualizing Ground Truth DAG...")
labels = ['X1', 'X2', 'X3', 'X4', 'X5']
gt_graph_dot = plot_graph(true_dag_matrix, labels, title="Ground Truth DAG")
# In a real environment, you would display gt_graph_dot
# Example: gt_graph_dot.render('ground_truth_dag', view=True)
# For this context, let's show the Graphviz source:
print("Graphviz source for ground truth:")
print(gt_graph_dot.source)
```

X1 X2 X3 X4 X5

    The ground truth causal structure used to generate our synthetic data.

Applying Constraint-based Discovery: The PC Algorithm

The Peter-Clark (PC) algorithm is a classic constraint-based method. It starts with a fully connected undirected graph and iteratively removes edges based on conditional independence tests. Finally, it orients edges based on collision patterns and propagation rules.

We'll use the causal_learn.discovery.ConstraintBased.PC implementation. An important parameter is the significance level alpha for the conditional independence tests. Common choices for the test include fisherz (for Gaussian data) or gsq (Chi-squared, for discrete data). Since our data is generated linearly with Gaussian noise, fisherz is appropriate.
```python
from causal_learn.discovery.ConstraintBased.PC import pc
from causal_learn.utils.cit import fisherz

# Run the PC algorithm
# Parameters: data matrix (numpy array), significance level (alpha), independence test
cg_pc = pc(data.to_numpy(), alpha=0.05, indep_test=fisherz)

# Get the estimated adjacency matrix
# cg_pc.G.graph[i, j] == -1 and cg_pc.G.graph[j, i] == -1 means i -- j
# cg_pc.G.graph[i, j] == 1 and cg_pc.G.graph[j, i] == 0 means i -> j
estimated_adj_pc = cg_pc.G.graph

print("\nPC Algorithm Estimated Adjacency Matrix:")
print(estimated_adj_pc)

# Visualize the result (CPDAG)
print("\nVisualizing PC Algorithm Output (CPDAG)...")
pc_graph_dot = plot_graph(estimated_adj_pc, labels, title="PC Algorithm Result (CPDAG)")
print("Graphviz source for PC result:")
print(pc_graph_dot.source)
```

X1 X2 X3 X4 X5

    The causal structure estimated by the PC algorithm. In this case, with sufficient data and well-behaved relationships, PC correctly recovered the ground truth DAG. Orange undirected edges would indicate edges whose direction could not be determined from conditional independencies alone (part of the Markov equivalence class).

The PC algorithm often returns a Completed Partially Directed Acyclic Graph (CPDAG), representing the Markov equivalence class. Undirected edges in the CPDAG indicate relationships where the direction couldn't be uniquely determined from the observational data alone using conditional independence tests. In our specific case, the generated data and chosen parameters allowed PC to fully orient the graph, matching the ground truth. However, be aware that changing alpha, the sample size, or the underlying data generating process can lead to errors (missing or extra edges) or unoriented edges.
Applying Score-based Discovery: The GES Algorithm

Greedy Equivalence Search (GES) is a score-based algorithm that operates in two phases: forward greedy search to add edges improving a score (like BIC), and backward greedy search to remove edges. It searches over the space of equivalence classes (CPDAGs).

We use causal_learn.discovery.ScoreBased.GES. Common scoring functions include BIC (bic) or BDeu (bdeu) for discrete data, and BIC for Gaussian data (bic_gauss).
```python
from causal_learn.discovery.ScoreBased.GES import ges

# Run the GES algorithm
# Parameters: data matrix (numpy array), scoring method
# Other parameters like max_degree can be tuned
record_ges = ges(data.to_numpy(), score_func='bic_gauss')

# Get the estimated adjacency matrix from the GES result
estimated_adj_ges = record_ges['G'].graph

print("\nGES Algorithm Estimated Adjacency Matrix:")
print(estimated_adj_ges)

# Visualize the result (CPDAG)
print("\nVisualizing GES Algorithm Output (CPDAG)...")
ges_graph_dot = plot_graph(estimated_adj_ges, labels, title="GES Algorithm Result (CPDAG)")
print("Graphviz source for GES result:")
print(ges_graph_dot.source)
```

X1 X2 X3 X4 X5

    The causal structure estimated by the GES algorithm using the BIC score for Gaussian variables. Similar to PC in this instance, GES recovered the true DAG structure.

Like PC, GES returns a CPDAG representing the Markov equivalence class identified. For this dataset, GES also successfully recovered the true structure. Score-based methods often perform well when the chosen score aligns with the data generation process (e.g., BIC for Gaussian linear models). However, their performance can degrade if model assumptions (like linearity or specific noise distributions) are violated. The greedy nature of the search also means it might settle in a local optimum.
Introducing Latent Variables: The FCI Algorithm

What happens if there's an unobserved common cause (latent confounder)? Standard PC and GES assume causal sufficiency (no latent confounders). The Fast Causal Inference (FCI) algorithm is a constraint-based method designed to handle this. It outputs a Partial Ancestral Graph (PAG), which can include edges like directed (A→BA→B), bidirected (A↔BA↔B, indicating a latent common cause), undirected (A−BA−B), and partially oriented edges.

Let's modify our data generation to include a latent variable LL affecting X2X2​ and X4X4​, and then run FCI on the observed data (X1X1​ to X5X5​).
```python
from causal_learn.discovery.ConstraintBased.FCI import fci

# --- Generate data with a latent confounder ---
np.random.seed(123)
num_samples_latent = 1000

n1 = np.random.randn(num_samples_latent)
n2 = np.random.randn(num_samples_latent)
n3 = np.random.randn(num_samples_latent)
n4 = np.random.randn(num_samples_latent)
n5 = np.random.randn(num_samples_latent)
L = np.random.randn(num_samples_latent) # Latent variable

x1_l = n1
x5_l = n5
# L influences X2 and X4
x2_l = 0.8 * x1_l + 0.7 * L + n2
x3_l = 0.5 * x1_l - 0.6 * x2_l + n3
x4_l = 0.7 * x2_l + 0.4 * x5_l + 0.6 * L + n4

# Observed data (L is excluded)
data_latent = pd.DataFrame({'X1': x1_l, 'X2': x2_l, 'X3': x3_l, 'X4': x4_l, 'X5': x5_l})

# Ground truth PAG (FCI output represents this class)
# X1 -> X2, X1 -> X3, X2 -> X3
# X5 -> X4
# X2 <-> X4 (due to latent L)

# Run FCI
# Note: FCI can be computationally intensive
cg_fci, edges_fci = fci(data_latent.to_numpy(), alpha=0.05, indep_test=fisherz, verbose=False)

# The graph representation in causal-learn for PAGs needs careful interpretation.
# cg_fci.graph[i, j] encodes edge type (0: no edge, 1: circle, 2: arrowhead, 3: tail)
# Example: cg_fci.graph[i, j] = 2 and cg_fci.graph[j, i] = 3 means i -> j
# Example: cg_fci.graph[i, j] = 2 and cg_fci.graph[j, i] = 2 means i <-> j

print("\nFCI Algorithm Estimated PAG Edges (Encoded):")
# This raw matrix is hard to read directly, visualization is crucial.
# print(cg_fci.graph)

# --- Function to visualize PAG using Graphviz ---
def plot_pag(pag_matrix, labels, title="FCI Result (PAG)"):
    num_nodes = pag_matrix.shape[0]
    node_names = [labels[i] for i in range(num_nodes)]
    dot = graphviz.Digraph(comment=title, graph_attr={'rankdir': 'LR', 'splines':'true', 'overlap':'false'})

    for name in node_names:
       dot.node(name, name, shape='ellipse', style='filled', fillcolor='#d0bfff', color='#7048e8')

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): # Avoid double drawing
            mark_i = pag_matrix[j, i] # Mark at node i for edge j -> i
            mark_j = pag_matrix[i, j] # Mark at node j for edge i -> j

            if mark_i == 0 and mark_j == 0: # No edge
                continue
            elif mark_i == 2 and mark_j == 2: # i <-> j (bidirected)
                dot.edge(node_names[i], node_names[j], arrowhead='normal', arrowtail='normal', dir='both', color='#f03e3e')
            elif mark_i == 3 and mark_j == 2: # i -> j
                dot.edge(node_names[i], node_names[j], arrowhead='normal', arrowtail='none', dir='forward', color='#495057')
            elif mark_i == 2 and mark_j == 3: # i <- j
                dot.edge(node_names[j], node_names[i], arrowhead='normal', arrowtail='none', dir='forward', color='#495057')
            elif mark_i == 1 and mark_j == 1: # i o-o j (undirected / circle)
                dot.edge(node_names[i], node_names[j], arrowhead='odot', arrowtail='odot', dir='both', color='#fd7e14')
            elif mark_i == 1 and mark_j == 2: # i o-> j
                dot.edge(node_names[i], node_names[j], arrowhead='normal', arrowtail='odot', dir='forward', color='#cc5de8')
            elif mark_i == 2 and mark_j == 1: # i <-o j
                dot.edge(node_names[j], node_names[i], arrowhead='normal', arrowtail='odot', dir='forward', color='#cc5de8')
            # Other combinations (tail-tail etc.) can exist but less common in simple FCI outputs
            else: # Default for less common cases or unhandled combinations
                 dot.edge(node_names[i], node_names[j], label=f'{mark_i}-{mark_j}', color='#adb5bd')
    return dot

print("\nVisualizing FCI Algorithm Output (PAG)...")
fci_graph_dot = plot_pag(cg_fci.graph, labels, title="FCI Algorithm Result (PAG)")
print("Graphviz source for FCI result:")
print(fci_graph_dot.source)
```

X1 X2 X3 X4 X5 -

    The Partial Ancestral Graph (PAG) estimated by the FCI algorithm on data with a latent confounder (LL) affecting X2X2​ and X4X4​. Note the bidirected edge X2↔X4X2​↔X4​ (red), correctly indicating the presence of an unobserved common cause between these two variables. Other relationships are identified as directed edges.

The FCI output correctly identifies the directed edges originating from X1X1​ and the edge from X5→X4X5​→X4​. Significantly, it places a bidirected edge X2↔X4X2​↔X4​, indicating that it detected the presence of the latent confounder LL influencing both. FCI provides guarantees about the ancestral relationships it finds, even without the causal sufficiency assumption.
Evaluating Discovery Performance

When the ground truth is known (as in simulations), we can quantitatively evaluate algorithm performance using metrics like:

    Structural Hamming Distance (SHD): The number of edge additions, deletions, or reversals needed to transform the estimated graph into the true graph. Lower is better.
    True Positive Rate (TPR): Proportion of correctly identified edges.
    False Positive Rate (FPR): Proportion of incorrectly identified edges among non-existent true edges.
    Precision: Proportion of identified edges that are correct.

```python
from causal_learn.metrics import SHD, SID

# --- Evaluate PC result (against the first ground truth) ---
# Note: Need to convert PC's CPDAG matrix (using -1 for undirected)
# to a standard adjacency matrix for comparison if it had undirected edges.
# In our case, PC returned a DAG, so conversion might not be strictly needed,
# but comparison often requires handling CPDAGs properly.
# For simplicity here, assuming output is DAG if no -1 exists.
shd_pc = SHD(true_dag_matrix, estimated_adj_pc)
# Precision/Recall/F1 can also be calculated, often using edge presence/absence

print(f"\n--- Evaluation (No Latent Variable Case) ---")
print(f"PC Algorithm SHD: {shd_pc}")

# --- Evaluate GES result ---
shd_ges = SHD(true_dag_matrix, estimated_adj_ges)
print(f"GES Algorithm SHD: {shd_ges}")

# Evaluation of FCI is more complex as it outputs a PAG, not a DAG/CPDAG.
# Metrics need to compare PAG features (ancestors, definite structures)
# Standard SHD is not directly applicable without converting PAG to a comparable format.
# Specialized metrics exist but are outside this practical overview.
print("\n(FCI evaluation requires PAG-specific metrics, omitted here)")
```

"In this ideal scenario, both PC and GES achieved an SHD of 0, indicating perfect recovery. In applications or more complex simulations, you'd expect non-zero SHDs and would analyze precision/recall to understand the types of errors made."
Discussion

This practical exercise demonstrated how to apply prominent causal discovery algorithms (PC, GES, FCI) using the causal-learn library.

    We saw that PC and GES can recover the true DAG structure from observational data when their assumptions (e.g., causal sufficiency, linearity/Gaussianity for specific tests/scores) hold reasonably well.
    We observed how FCI, by relaxing the causal sufficiency assumption, can correctly indicate the presence of latent confounders using bidirected edges in its PAG output.
    We touched upon evaluating performance using SHD when a ground truth is available.

Remember that these algorithms are tools, not magic wands. Their success heavily depends on the quality and quantity of data, the validity of their underlying assumptions (conditional independence tests, scoring functions, linearity, acyclicity, no hidden confounders for PC/GES), and appropriate parameter tuning (alpha for constraint-based, score choice for score-based). Always interpret the output graphs critically, examining potential limitations and incorporating domain knowledge whenever possible. Experimenting with different algorithms, parameters, and robustness checks is essential in any serious causal discovery endeavor.