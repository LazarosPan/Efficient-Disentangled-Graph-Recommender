Graph Convolutional Networks (GCN) (https://apxml.com/courses/graph-neural-networks-gnns/chapter-2-advanced-gnn-architectures/gcn-exploration)

An examination of specific advanced Graph Neural Network architectures is presented. While many innovative designs exist, the Graph Convolutional Network (GCN) remains a foundation and a frequent starting point for understanding more complex models. Its spectral roots are formally analyzed, connecting them to its widely used spatial implementation based on message passing.
From Spectral Filters to Spatial Aggregation

Recall from Chapter 1 that spectral graph convolutions operate in the Fourier domain, defined by the eigenvectors of the graph Laplacian LL. A spectral filter gθgθ​ applied to a graph signal xx is defined as:
gθ∗x=Ugθ(Λ)UTx
gθ​∗x=Ugθ​(Λ)UTx

Here, UU contains the eigenvectors of L=D−AL=D−A, ΛΛ is the diagonal matrix of eigenvalues, and gθ(Λ)gθ​(Λ) applies the filter function element-wise to the eigenvalues. While theoretically elegant, this formulation suffers from two main drawbacks:

    Computational Cost: Calculating the eigen-decomposition of LL for large graphs is computationally expensive, often O(N3)O(N3) where NN is the number of nodes.
    Non-Localization: The eigenvectors UU are generally dense, meaning the filter is not localized in the vertex domain; changing a single node's feature can influence the output across the entire graph.

To address these issues, approximations using polynomials in the Laplacian were proposed. ChebNet, for instance, uses Chebyshev polynomials TkTk​ to approximate the filter:
gθ(Λ)≈∑k=0KθkTk(Λ~)
gθ​(Λ)≈k=0∑K​θk​Tk​(Λ~)

where Λ~Λ~ is a rescaled version of ΛΛ to lie within [−1,1][−1,1], the domain where Chebyshev polynomials are defined. This polynomial approximation makes the filter KK-localized, meaning it only depends on the KK-hop neighborhood of a node, and avoids the explicit computation of eigenvectors.
The GCN Simplification

The Graph Convolutional Network, as introduced by Kipf and Welling (2017), can be understood as a specific, highly effective simplification of this spectral approach. It essentially makes two important approximations:

    Linear Filter: It restricts the polynomial filter to K=1K=1. This implies the filter examines only the immediate (1-hop) neighborhood. The filter function becomes linear with respect to the Laplacian's eigenvalues: gθ(Λ)≈θ0I+θ1Λgθ​(Λ)≈θ0​I+θ1​Λ.
    Parameter Reduction: It further simplifies by setting θ0=−θ1=θθ0​=−θ1​=θ, reducing the filter to gθ(Λ)≈θ(I−Λ)gθ​(Λ)≈θ(I−Λ). However, practical GCNs use a slightly different formulation based on a renormalized adjacency matrix.

The main step in the practical GCN formulation involves a renormalization trick. Instead of directly using the Laplacian L=I−D−1/2AD−1/2L=I−D−1/2AD−1/2 (normalized Laplacian), GCN employs a modified adjacency matrix A^A^:
A^=D~−1/2A~D~−1/2
A^=D~−1/2A~D~−1/2

where A~=A+IA~=A+I is the adjacency matrix with self-loops added, and D~D~ is the diagonal degree matrix of A~A~ (i.e., D~ii=∑jA~ijD~ii​=∑j​A~ij​).

Why this specific form?

    Adding Self-Loops (A+IA+I): Ensures that a node's own features are included in the aggregation process during message passing. Without this, a node's updated representation would solely depend on its neighbors.
    Symmetric Normalization (D~−1/2…D~−1/2D~−1/2…D~−1/2): This normalization prevents the scale of node feature vectors from changing drastically based on node degrees during aggregation. Multiplying by AA would sum neighbor features, causing high-degree nodes to have much larger feature values. Dividing by D~D~ (as in D~−1A~D~−1A~) averages neighbor features but can lead to other numerical instabilities. The symmetric normalization D~−1/2A~D~−1/2D~−1/2A~D~−1/2 balances these effects and has favorable spectral properties. Its eigenvalues are bounded within [0,2][0,2] when derived from I+D−1/2AD−1/2I+D−1/2AD−1/2 (assuming AA has self-loops or using A+IA+I originally), which helps stabilize training, particularly in deeper networks.

The GCN Layer Propagation Rule

Combining these elements, the layer-wise propagation rule for a GCN takes a remarkably simple form. Given node features H(l)∈RN×FlH(l)∈RN×Fl​ at layer ll (where NN is the number of nodes and FlFl​ is the feature dimension), the features H(l+1)∈RN×Fl+1H(l+1)∈RN×Fl+1​ at the next layer are computed as:
H(l+1)=σ(A^H(l)W(l))
H(l+1)=σ(A^H(l)W(l))

Let's break this down:

    H(l)H(l): The matrix of node embeddings from the previous layer (or initial node features for l=0l=0).
    W(l)∈RFl×Fl+1W(l)∈RFl​×Fl+1​: A layer-specific trainable weight matrix. This matrix transforms the features of each node.
    A^∈RN×NA^∈RN×N: The pre-computed, normalized adjacency matrix with self-loops. Multiplication by A^A^ performs the core neighborhood aggregation. Specifically, (A^X)i=∑j∈N(i)∪{i}1d~id~jXj(A^X)i​=∑j∈N(i)∪{i}​d~i​d~j​
    ​1​Xj​, where X=H(l)W(l)X=H(l)W(l) and d~id~i​ is the degree of node ii in A~A~.
    σ(⋅)σ(⋅): An element-wise non-linear activation function, such as ReLU or ELU.

Multiple GCN layers can be stacked to allow information to propagate across larger distances in the graph. A typical two-layer GCN for semi-supervised node classification might look like:
Z=softmax(A^ ReLU(A^XW(0))W(1))
Z=softmax(A^ ReLU(A^XW(0))W(1))

where XX are the input features, W(0)W(0) and W(1)W(1) are the weight matrices, and ZZ contains the output class probabilities for each node.
GCN as Message Passing

The GCN propagation rule fits neatly into the message passing framework discussed in Chapter 1. For a single node ii, the update can be seen as:

    Transformation: Each node jj (including ii itself due to the self-loop in A~A~) transforms its feature vector hj(l)hj(l)​ using the weight matrix: mj(l)=hj(l)W(l)mj(l)​=hj(l)​W(l).
    Aggregation: Node ii aggregates the transformed messages from its neighbors (and itself), weighted by the normalization constants from A^A^:
    ai(l)=∑j∈N(i)∪{i}1d~id~jmj(l)
    ai(l)​=j∈N(i)∪{i}∑​d~i​d~j​
    ​1​mj(l)​ This aggregation is implicitly performed by the matrix multiplication A^(H(l)W(l))A^(H(l)W(l)).
    Update: Apply the non-linear activation function:
    hi(l+1)=σ(ai(l))
    hi(l+1)​=σ(ai(l)​)

This perspective highlights that GCN performs a specific, fixed form of neighborhood aggregation determined by the graph structure (A^A^) followed by a shared linear transformation (W(l)W(l)) and non-linearity.
Neighbors of Node i (Layer l) Node i (Layer l) Transform (Wₗ) Aggregate (Weighted Sum via Â) Update (σ) hⱼ1 hⱼ1 * Wₗ hⱼ2 hⱼ2 * Wₗ ... hᵢ hᵢ * Wₗ ∑ neighbors + self 1/√(di*dj1) 1/√(di*dj2) 1/√(di*di) ... hᵢ (l+1) σ(.)

    Message passing view of a GCN layer update for node ii. Features from neighbors and the node itself (hj,hihj​,hi​) are transformed by W(l)W(l), aggregated using weights derived from the normalized adjacency matrix A^A^, and then passed through a non-linearity σσ.

Strengths and Limitations

GCNs gained popularity due to their simplicity, efficiency (compared to earlier spectral methods), and strong performance on benchmark tasks like semi-supervised node classification on citation networks (Cora, Citeseer, Pubmed). They provide an effective baseline model.

However, GCNs have limitations:

    Limited Expressivity: The fixed aggregation scheme based on A^A^ treats all neighbors equally (after normalization) and limits the model's ability to distinguish certain graph structures, as discussed in relation to the WL test in Chapter 1.
    Oversmoothing: Stacking many GCN layers tends to make node representations converge towards a common value, losing discriminative power. The repeated averaging across neighborhoods smooths out node features.
    Homophily Assumption: GCNs implicitly work best under the assumption of homophily, where connected nodes tend to have similar features or labels. The averaging mechanism reinforces this. Performance can degrade on graphs with significant heterophily (where connected nodes are often dissimilar).
    Inability to Handle Weighted Edges Natively: The standard GCN formulation uses the unweighted adjacency matrix AA. While modifications exist, the base model doesn't inherently incorporate edge weights or features into the aggregation weighting.

These limitations motivate the development of more sophisticated architectures. For instance, Graph Attention Networks (GATs), discussed next, introduce learnable attention mechanisms to assign different importance weights to neighbors during aggregation, directly addressing the fixed weighting scheme of GCNs. Other architectures tackle oversmoothing or adapt GNNs for different graph types and tasks, as we will discuss throughout this course. Understanding the spectral origins and practical message-passing implementation of GCN provides a foundation for appreciating these subsequent advancements.