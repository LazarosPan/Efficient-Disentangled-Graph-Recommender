torch_geometric.nn

Contents

    Convolutional Layers

    Aggregation Operators

    Attention

    Normalization Layers

    Pooling Layers

    Unpooling Layers

    Models

    KGE Models

    Encodings

    Functional

    Dense Convolutional Layers

    Dense Pooling Layers

    Model Transformations

    DataParallel Layers

    Model Hub

    Model Summary

class Sequential(input_args: str, modules: List[Union[Tuple[Callable, str], Callable]])[source]

    An extension of the torch.nn.Sequential container in order to define a sequential GNN model.

    Since GNN operators take in multiple input arguments, torch_geometric.nn.Sequential additionally expects both global input arguments, and function header definitions of individual operators. If omitted, an intermediate module will operate on the output of its preceding module:

    from torch.nn import Linear, ReLU
    from torch_geometric.nn import Sequential, GCNConv

    model = Sequential('x, edge_index', [
        (GCNConv(in_channels, 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(64, 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        Linear(64, out_channels),
    ])

    Here, 'x, edge_index' defines the input arguments of model, and 'x, edge_index -> x' defines the function header, i.e. input arguments and return types of GCNConv.

    In particular, this also allows to create more sophisticated models, such as utilizing JumpingKnowledge:

    from torch.nn import Linear, ReLU, Dropout
    from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
    from torch_geometric.nn import global_mean_pool

    model = Sequential('x, edge_index, batch', [
        (Dropout(p=0.5), 'x -> x'),
        (GCNConv(dataset.num_features, 64), 'x, edge_index -> x1'),
        ReLU(inplace=True),
        (GCNConv(64, 64), 'x1, edge_index -> x2'),
        ReLU(inplace=True),
        (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
        (JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
        (global_mean_pool, 'x, batch -> x'),
        Linear(2 * 64, dataset.num_classes),
    ])

    Parameters:

            input_args (str) – The input arguments of the model.

            modules ([(Callable, str) or Callable]) – A list of modules (with optional function header definitions). Alternatively, an OrderedDict of modules (and function header definitions) can be passed.

class Linear(in_channels: int, out_channels: int, bias: bool = True, weight_initializer: Optional[str] = None, bias_initializer: Optional[str] = None)[source]

    Applies a linear transformation to the incoming data.

    In contrast to torch.nn.Linear, it supports lazy initialization and customizable weight and bias initialization.

    Parameters:

            in_channels (int) – Size of each input sample. Will be initialized lazily in case it is given as -1.

            out_channels (int) – Size of each output sample.

            bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)

            weight_initializer (str, optional) – The initializer for the weight matrix ("glorot", "uniform", "kaiming_uniform" or None). If set to None, will match default weight initialization of torch.nn.Linear. (default: None)

            bias_initializer (str, optional) – The initializer for the bias vector ("zeros" or None). If set to None, will match default bias initialization of torch.nn.Linear. (default: None)

    Shapes:

            input: features

            output: features

    reset_parameters()[source]

        Resets all learnable parameters of the module.

    forward(x: Tensor) → Tensor[source]

        Forward pass.

        Parameters:

            x (torch.Tensor) – The input features.
        Return type:

            Tensor

class HeteroLinear(in_channels: int, out_channels: int, num_types: int, is_sorted: bool = False, **kwargs)[source]

    Applies separate linear transformations to the incoming data according to types.

    For type , it computes

    It supports lazy initialization and customizable weight and bias initialization.

    Parameters:

            in_channels (int) – Size of each input sample. Will be initialized lazily in case it is given as -1.

            out_channels (int) – Size of each output sample.

            num_types (int) – The number of types.

            is_sorted (bool, optional) – If set to True, assumes that type_vec is sorted. This avoids internal re-sorting of the data and can improve runtime and memory efficiency. (default: False)

            **kwargs (optional) – Additional arguments of torch_geometric.nn.Linear.

    Shapes:

            input: features , type vector

            output: features

    reset_parameters()[source]

        Resets all learnable parameters of the module.

    forward(x: Tensor, type_vec: Tensor) → Tensor[source]

        The forward pass.

        Parameters:

                x (torch.Tensor) – The input features.

                type_vec (torch.Tensor) – A vector that maps each entry to a type.

        Return type:

            Tensor

class HeteroDictLinear(in_channels: Union[int, Dict[Any, int]], out_channels: int, types: Optional[Any] = None, **kwargs)[source]

    Applies separate linear transformations to the incoming data dictionary.

    For key , it computes

    It supports lazy initialization and customizable weight and bias initialization.

    Parameters:

            in_channels (int or Dict[Any, int]) – Size of each input sample. If passed an integer, types will be a mandatory argument. initialized lazily in case it is given as -1.

            out_channels (int) – Size of each output sample.

            types (List[Any], optional) – The keys of the input dictionary. (default: None)

            **kwargs (optional) – Additional arguments of torch_geometric.nn.Linear.

    reset_parameters()[source]

        Resets all learnable parameters of the module.

    forward(x_dict: Dict[str, Tensor]) → Dict[str, Tensor][source]

        Forward pass.

        Parameters:

            x_dict (Dict[Any, torch.Tensor]) – A dictionary holding input features for each individual type.
        Return type:

            Dict[str, Tensor]

Convolutional Layers

MessagePassing
	

Base class for creating message passing layers.

SimpleConv
	

A simple message passing operator that performs (non-trainable) propagation.

GCNConv
	

The graph convolutional operator from the "Semi-supervised Classification with Graph Convolutional Networks" paper.

ChebConv
	

The chebyshev spectral graph convolutional operator from the "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" paper.

SAGEConv
	

The GraphSAGE operator from the "Inductive Representation Learning on Large Graphs" paper.

CuGraphSAGEConv
	

The GraphSAGE operator from the "Inductive Representation Learning on Large Graphs" paper.

GraphConv
	

The graph neural network operator from the "Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks" paper.

GravNetConv
	

The GravNet operator from the "Learning Representations of Irregular Particle-detector Geometry with Distance-weighted Graph Networks" paper, where the graph is dynamically constructed using nearest neighbors.

GatedGraphConv
	

The gated graph convolution operator from the "Gated Graph Sequence Neural Networks" paper.

ResGatedGraphConv
	

The residual gated graph convolutional operator from the "Residual Gated Graph ConvNets" paper.

GATConv
	

The graph attentional operator from the "Graph Attention Networks" paper.

CuGraphGATConv
	

The graph attentional operator from the "Graph Attention Networks" paper.

FusedGATConv
	

The fused graph attention operator from the "Understanding GNN Computational Graph: A Coordinated Computation, IO, and Memory Perspective" paper.

GATv2Conv
	

The GATv2 operator from the "How Attentive are Graph Attention Networks?" paper, which fixes the static attention problem of the standard GATConv layer.

TransformerConv
	

The graph transformer operator from the "Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification" paper.

AGNNConv
	

The graph attentional propagation layer from the "Attention-based Graph Neural Network for Semi-Supervised Learning" paper.

TAGConv
	

The topology adaptive graph convolutional networks operator from the "Topology Adaptive Graph Convolutional Networks" paper.

GINConv
	

The graph isomorphism operator from the "How Powerful are Graph Neural Networks?" paper.

GINEConv
	

The modified GINConv operator from the "Strategies for Pre-training Graph Neural Networks" paper.

ARMAConv
	

The ARMA graph convolutional operator from the "Graph Neural Networks with Convolutional ARMA Filters" paper.

SGConv
	

The simple graph convolutional operator from the "Simplifying Graph Convolutional Networks" paper.

SSGConv
	

The simple spectral graph convolutional operator from the "Simple Spectral Graph Convolution" paper.

APPNP
	

The approximate personalized propagation of neural predictions layer from the "Predict then Propagate: Graph Neural Networks meet Personalized PageRank" paper.

MFConv
	

The graph neural network operator from the "Convolutional Networks on Graphs for Learning Molecular Fingerprints" paper.

RGCNConv
	

The relational graph convolutional operator from the "Modeling Relational Data with Graph Convolutional Networks" paper.

FastRGCNConv
	

See RGCNConv.

CuGraphRGCNConv
	

The relational graph convolutional operator from the "Modeling Relational Data with Graph Convolutional Networks" paper.

RGATConv
	

The relational graph attentional operator from the "Relational Graph Attention Networks" paper.

SignedConv
	

The signed graph convolutional operator from the "Signed Graph Convolutional Network" paper.

DNAConv
	

The dynamic neighborhood aggregation operator from the "Just Jump: Towards Dynamic Neighborhood Aggregation in Graph Neural Networks" paper.

PointNetConv
	

The PointNet set layer from the "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" and "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" papers.

GMMConv
	

The gaussian mixture model convolutional operator from the "Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs" paper.

SplineConv
	

The spline-based convolutional operator from the "SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels" paper.

NNConv
	

The continuous kernel-based convolutional operator from the "Neural Message Passing for Quantum Chemistry" paper.

CGConv
	

The crystal graph convolutional operator from the "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties" paper.

EdgeConv
	

The edge convolutional operator from the "Dynamic Graph CNN for Learning on Point Clouds" paper.

DynamicEdgeConv
	

The dynamic edge convolutional operator from the "Dynamic Graph CNN for Learning on Point Clouds" paper (see torch_geometric.nn.conv.EdgeConv), where the graph is dynamically constructed using nearest neighbors in the feature space.

XConv
	

The convolutional operator on -transformed points from the "PointCNN: Convolution On X-Transformed Points" paper.

PPFConv
	

The PPFNet operator from the "PPFNet: Global Context Aware Local Features for Robust 3D Point Matching" paper.

FeaStConv
	

The (translation-invariant) feature-steered convolutional operator from the "FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis" paper.

PointTransformerConv
	

The Point Transformer layer from the "Point Transformer" paper.

HypergraphConv
	

The hypergraph convolutional operator from the "Hypergraph Convolution and Hypergraph Attention" paper.

LEConv
	

The local extremum graph neural network operator from the "ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations" paper.

PNAConv
	

The Principal Neighbourhood Aggregation graph convolution operator from the "Principal Neighbourhood Aggregation for Graph Nets" paper.

ClusterGCNConv
	

The ClusterGCN graph convolutional operator from the "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks" paper.

GENConv
	

The GENeralized Graph Convolution (GENConv) from the "DeeperGCN: All You Need to Train Deeper GCNs" paper.

GCN2Conv
	

The graph convolutional operator with initial residual connections and identity mapping (GCNII) from the "Simple and Deep Graph Convolutional Networks" paper.

PANConv
	

The path integral based convolutional operator from the "Path Integral Based Convolution and Pooling for Graph Neural Networks" paper.

WLConv
	

The Weisfeiler Lehman (WL) operator from the "A Reduction of a Graph to a Canonical Form and an Algebra Arising During this Reduction" paper.

WLConvContinuous
	

The Weisfeiler Lehman operator from the "Wasserstein Weisfeiler-Lehman Graph Kernels" paper.

FiLMConv
	

The FiLM graph convolutional operator from the "GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation" paper.

SuperGATConv
	

The self-supervised graph attentional operator from the "How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision" paper.

FAConv
	

The Frequency Adaptive Graph Convolution operator from the "Beyond Low-Frequency Information in Graph Convolutional Networks" paper.

EGConv
	

The Efficient Graph Convolution from the "Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions" paper.

PDNConv
	

The pathfinder discovery network convolutional operator from the "Pathfinder Discovery Networks for Neural Message Passing" paper.

GeneralConv
	

A general GNN layer adapted from the "Design Space for Graph Neural Networks" paper.

HGTConv
	

The Heterogeneous Graph Transformer (HGT) operator from the "Heterogeneous Graph Transformer" paper.

HEATConv
	

The heterogeneous edge-enhanced graph attentional operator from the "Heterogeneous Edge-Enhanced Graph Attention Network For Multi-Agent Trajectory Prediction" paper.

HeteroConv
	

A generic wrapper for computing graph convolution on heterogeneous graphs.

HANConv
	

The Heterogenous Graph Attention Operator from the "Heterogenous Graph Attention Network" paper.

LGConv
	

The Light Graph Convolution (LGC) operator from the "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" paper.

PointGNNConv
	

The PointGNN operator from the "Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud" paper.

GPSConv
	

The general, powerful, scalable (GPS) graph transformer layer from the "Recipe for a General, Powerful, Scalable Graph Transformer" paper.

AntiSymmetricConv
	

The anti-symmetric graph convolutional operator from the "Anti-Symmetric DGN: a stable architecture for Deep Graph Networks" paper.

DirGNNConv
	

A generic wrapper for computing graph convolution on directed graphs as described in the "Edge Directionality Improves Learning on Heterophilic Graphs" paper.

MixHopConv
	

The Mix-Hop graph convolutional operator from the "MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing" paper.

MeshCNNConv
	

The convolutional layer introduced by the paper "MeshCNN: A Network With An Edge".
Aggregation Operators

Aggregation functions play an important role in the message passing framework and the readout functions of Graph Neural Networks. Specifically, many works in the literature (Hamilton et al. (2017), Xu et al. (2018), Corso et al. (2020), Li et al. (2020), Tailor et al. (2021)) demonstrate that the choice of aggregation functions contributes significantly to the representational power and performance of the model. For example, mean aggregation captures the distribution (or proportions) of elements, max aggregation proves to be advantageous to identify representative elements, and sum aggregation enables the learning of structural graph properties (Xu et al. (2018)). Recent works also show that using multiple aggregations (Corso et al. (2020), Tailor et al. (2021)) and learnable aggregations (Li et al. (2020)) can potentially provide substantial improvements. Another line of research studies optimization-based and implicitly-defined aggregations (Bartunov et al. (2022)). Furthermore, an interesting discussion concerns the trade-off between representational power (usually gained through learnable functions implemented as neural networks) and the formal property of permutation invariance (Buterez et al. (2022)).

To facilitate further experimentation and unify the concepts of aggregation within GNNs across both MessagePassing and global readouts, we have made the concept of Aggregation a first-class principle in PyG. As of now, PyG provides support for various aggregations — from rather simple ones (e.g., mean, max, sum), to advanced ones (e.g., median, var, std), learnable ones (e.g., SoftmaxAggregation, PowerMeanAggregation, SetTransformerAggregation), and exotic ones (e.g., MLPAggregation, LSTMAggregation, SortAggregation, EquilibriumAggregation):

from torch_geometric.nn import aggr

# Simple aggregations:
mean_aggr = aggr.MeanAggregation()
max_aggr = aggr.MaxAggregation()

# Advanced aggregations:
median_aggr = aggr.MedianAggregation()

# Learnable aggregations:
softmax_aggr = aggr.SoftmaxAggregation(learn=True)
powermean_aggr = aggr.PowerMeanAggregation(learn=True)

# Exotic aggregations:
lstm_aggr = aggr.LSTMAggregation(in_channels=..., out_channels=...)
sort_aggr = aggr.SortAggregation(k=4)

We can then easily apply these aggregations over a batch of sets of potentially varying size. For this, an index vector defines the mapping from input elements to their location in the output:

# Feature matrix holding 1000 elements with 64 features each:
x = torch.randn(1000, 64)

# Randomly assign elements to 100 sets:
index = torch.randint(0, 100, (1000, ))

output = mean_aggr(x, index)  #  Output shape: [100, 64]

Notably, all aggregations share the same set of forward arguments, as described in detail in the torch_geometric.nn.aggr.Aggregation base class.

Each of the provided aggregations can be used within MessagePassing as well as for hierarchical/global pooling to obtain graph-level representations:

import torch
from torch_geometric.nn import MessagePassing

class MyConv(MessagePassing):
    def __init__(self, ...):
        # Use a learnable softmax neighborhood aggregation:
        super().__init__(aggr=aggr.SoftmaxAggregation(learn=True))

   def forward(self, x, edge_index):
       ....


class MyGNN(torch.nn.Module)
    def __init__(self, ...):
        super().__init__()

        self.conv = MyConv(...)
        # Use a global sort aggregation:
        self.global_pool = aggr.SortAggregation(k=4)
        self.classifier = torch.nn.Linear(...)

     def forward(self, x, edge_index, batch):
         x = self.conv(x, edge_index).relu()
         x = self.global_pool(x, batch)
         x = self.classifier(x)
         return x

In addition, the aggregation package of PyG introduces two new concepts: First, aggregations can be resolved from pure strings via a lookup table, following the design principles of the class-resolver library, e.g., by simply passing in "median" to the MessagePassing module. This will automatically resolve to the MedianAggregation class:

class MyConv(MessagePassing):
    def __init__(self, ...):
        super().__init__(aggr="median")

Secondly, multiple aggregations can be combined and stacked via the MultiAggregation module in order to enhance the representational power of GNNs (Corso et al. (2020), Tailor et al. (2021)):

class MyConv(MessagePassing):
    def __init__(self, ...):
        # Combines a set of aggregations and concatenates their results,
        # i.e. its output will be `[num_nodes, 3 * out_channels]` here.
        # Note that the interface also supports automatic resolution.
        super().__init__(aggr=aggr.MultiAggregation(
            ['mean', 'std', aggr.SoftmaxAggregation(learn=True)]))

Importantly, MultiAggregation provides various options to combine the outputs of its underlying aggregations (e.g., using concatenation, summation, attention, …) via its mode argument. The default mode performs concatenation ("cat"). For combining via attention, we need to additionally specify the in_channels out_channels, and num_heads:

multi_aggr = aggr.MultiAggregation(
    aggrs=['mean', 'std'],
    mode='attn',
    mode_kwargs=dict(in_channels=64, out_channels=64, num_heads=4),
)

If aggregations are given as a list, they will be automatically resolved to a MultiAggregation, e.g., aggr=['mean', 'std', 'median'].

Finally, we added full support for customization of aggregations into the SAGEConv layer — simply override its aggr argument and utilize the power of aggregation within your GNN.

Note

You can read more about the torch_geometric.nn.aggr package in this blog post.

Aggregation
	

An abstract base class for implementing custom aggregations.

MultiAggregation
	

Performs aggregations with one or more aggregators and combines aggregated results, as described in the "Principal Neighbourhood Aggregation for Graph Nets" and "Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions" papers.

SumAggregation
	

An aggregation operator that sums up features across a set of elements.

MeanAggregation
	

An aggregation operator that averages features across a set of elements.

MaxAggregation
	

An aggregation operator that takes the feature-wise maximum across a set of elements.

MinAggregation
	

An aggregation operator that takes the feature-wise minimum across a set of elements.

MulAggregation
	

An aggregation operator that multiples features across a set of elements.

VarAggregation
	

An aggregation operator that takes the feature-wise variance across a set of elements.

StdAggregation
	

An aggregation operator that takes the feature-wise standard deviation across a set of elements.

SoftmaxAggregation
	

The softmax aggregation operator based on a temperature term, as described in the "DeeperGCN: All You Need to Train Deeper GCNs" paper.

PowerMeanAggregation
	

The powermean aggregation operator based on a power term, as described in the "DeeperGCN: All You Need to Train Deeper GCNs" paper.

MedianAggregation
	

An aggregation operator that returns the feature-wise median of a set.

QuantileAggregation
	

An aggregation operator that returns the feature-wise -th quantile of a set .

LSTMAggregation
	

Performs LSTM-style aggregation in which the elements to aggregate are interpreted as a sequence, as described in the "Inductive Representation Learning on Large Graphs" paper.

GRUAggregation
	

Performs GRU aggregation in which the elements to aggregate are interpreted as a sequence, as described in the "Graph Neural Networks with Adaptive Readouts" paper.

Set2Set
	

The Set2Set aggregation operator based on iterative content-based attention, as described in the "Order Matters: Sequence to sequence for Sets" paper.

DegreeScalerAggregation
	

Combines one or more aggregators and transforms its output with one or more scalers as introduced in the "Principal Neighbourhood Aggregation for Graph Nets" paper.

SortAggregation
	

The pooling operator from the "An End-to-End Deep Learning Architecture for Graph Classification" paper, where node features are sorted in descending order based on their last feature channel.

GraphMultisetTransformer
	

The Graph Multiset Transformer pooling operator from the "Accurate Learning of Graph Representations with Graph Multiset Pooling" paper.

AttentionalAggregation
	

The soft attention aggregation layer from the "Graph Matching Networks for Learning the Similarity of Graph Structured Objects" paper.

EquilibriumAggregation
	

The equilibrium aggregation layer from the "Equilibrium Aggregation: Encoding Sets via Optimization" paper.

MLPAggregation
	

Performs MLP aggregation in which the elements to aggregate are flattened into a single vectorial representation, and are then processed by a Multi-Layer Perceptron (MLP), as described in the "Graph Neural Networks with Adaptive Readouts" paper.

DeepSetsAggregation
	

Performs Deep Sets aggregation in which the elements to aggregate are first transformed by a Multi-Layer Perceptron (MLP) , summed, and then transformed by another MLP , as suggested in the "Graph Neural Networks with Adaptive Readouts" paper.

SetTransformerAggregation
	

Performs "Set Transformer" aggregation in which the elements to aggregate are processed by multi-head attention blocks, as described in the "Graph Neural Networks with Adaptive Readouts" paper.

LCMAggregation
	

The Learnable Commutative Monoid aggregation from the "Learnable Commutative Monoids for Graph Neural Networks" paper, in which the elements are aggregated using a binary tree reduction with depth.

VariancePreservingAggregation
	

Performs the Variance Preserving Aggregation (VPA) from the "GNN-VPA: A Variance-Preserving Aggregation Strategy for Graph Neural Networks" paper.

PatchTransformerAggregation
	

Performs patch transformer aggregation in which the elements to aggregate are processed by multi-head attention blocks across patches, as described in the "Simplifying Temporal Heterogeneous Network for Continuous-Time Link Prediction" paper.
Attention

PerformerAttention
	

The linear scaled attention mechanism from the "Rethinking Attention with Performers" paper.

QFormer
	

The Querying Transformer (Q-Former) from "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" paper.

SGFormerAttention
	

The simple global attention mechanism from the "SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations" paper.

PolynormerAttention
	

The polynomial-expressive attention mechanism from the "Polynormer: Polynomial-Expressive Graph Transformer in Linear Time" paper.
Normalization Layers

BatchNorm
	

Applies batch normalization over a batch of features as described in the "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" paper.

HeteroBatchNorm
	

Applies batch normalization over a batch of heterogeneous features as described in the "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" paper.

InstanceNorm
	

Applies instance normalization over each individual example in a batch of node features as described in the "Instance Normalization: The Missing Ingredient for Fast Stylization" paper.

LayerNorm
	

Applies layer normalization over each individual example in a batch of features as described in the "Layer Normalization" paper.

HeteroLayerNorm
	

Applies layer normalization over each individual example in a batch of heterogeneous features as described in the "Layer Normalization" paper.

GraphNorm
	

Applies graph normalization over individual graphs as described in the "GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training" paper.

GraphSizeNorm
	

Applies Graph Size Normalization over each individual graph in a batch of node features as described in the "Benchmarking Graph Neural Networks" paper.

PairNorm
	

Applies pair normalization over node features as described in the "PairNorm: Tackling Oversmoothing in GNNs" paper.

MeanSubtractionNorm
	

Applies layer normalization by subtracting the mean from the inputs as described in the "Revisiting 'Over-smoothing' in Deep GCNs" paper.

MessageNorm
	

Applies message normalization over the aggregated messages as described in the "DeeperGCNs: All You Need to Train Deeper GCNs" paper.

DiffGroupNorm
	

The differentiable group normalization layer from the "Towards Deeper Graph Neural Networks with Differentiable Group Normalization" paper, which normalizes node features group-wise via a learnable soft cluster assignment.
Pooling Layers

global_add_pool
	

Returns batch-wise graph-level-outputs by adding node features across the node dimension.

global_mean_pool
	

Returns batch-wise graph-level-outputs by averaging node features across the node dimension.

global_max_pool
	

Returns batch-wise graph-level-outputs by taking the channel-wise maximum across the node dimension.

KNNIndex
	

A base class to perform fast -nearest neighbor search (-NN) via the faiss library.

L2KNNIndex
	

Performs fast -nearest neighbor search (-NN) based on the metric via the faiss library.

MIPSKNNIndex
	

Performs fast -nearest neighbor search (-NN) based on the maximum inner product via the faiss library.

ApproxL2KNNIndex
	

Performs fast approximate -nearest neighbor search (-NN) based on the the metric via the faiss library.

ApproxMIPSKNNIndex
	

Performs fast approximate -nearest neighbor search (-NN) based on the maximum inner product via the faiss library.

TopKPooling
	

pooling operator from the "Graph U-Nets", "Towards Sparse Hierarchical Graph Classifiers" and "Understanding Attention and Generalization in Graph Neural Networks" papers.

SAGPooling
	

The self-attention pooling operator from the "Self-Attention Graph Pooling" and "Understanding Attention and Generalization in Graph Neural Networks" papers.

EdgePooling
	

The edge pooling operator from the "Towards Graph Pooling by Edge Contraction" and "Edge Contraction Pooling for Graph Neural Networks" papers.

ClusterPooling
	

The cluster pooling operator from the "Edge-Based Graph Component Pooling" paper.

ASAPooling
	

The Adaptive Structure Aware Pooling operator from the "ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations" paper.

PANPooling
	

The path integral based pooling operator from the "Path Integral Based Convolution and Pooling for Graph Neural Networks" paper.

MemPooling
	

Memory based pooling layer from "Memory-Based Graph Networks" paper, which learns a coarsened graph representation based on soft cluster assignments.

max_pool
	

Pools and coarsens a graph given by the torch_geometric.data.Data object according to the clustering defined in cluster.

avg_pool
	

Pools and coarsens a graph given by the torch_geometric.data.Data object according to the clustering defined in cluster.

max_pool_x
	

Max-Pools node features according to the clustering defined in cluster.

max_pool_neighbor_x
	

Max pools neighboring node features, where each feature in data.x is replaced by the feature value with the maximum value from the central node and its neighbors.

avg_pool_x
	

Average pools node features according to the clustering defined in cluster.

avg_pool_neighbor_x
	

Average pools neighboring node features, where each feature in data.x is replaced by the average feature values from the central node and its neighbors.

graclus
	

A greedy clustering algorithm from the "Weighted Graph Cuts without Eigenvectors: A Multilevel Approach" paper of picking an unmarked vertex and matching it with one of its unmarked neighbors (that maximizes its edge weight).

voxel_grid
	

Voxel grid pooling from the, e.g., Dynamic Edge-Conditioned Filters in Convolutional Networks on Graphs paper, which overlays a regular grid of user-defined size over a point cloud and clusters all points within the same voxel.

fps
	

A sampling algorithm from the "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" paper, which iteratively samples the most distant point with regard to the rest points.

knn
	

Finds for each element in y the k nearest points in x.

knn_graph
	

Computes graph edges to the nearest k points.

approx_knn
	

Finds for each element in y the k approximated nearest points in x.

approx_knn_graph
	

Computes graph edges to the nearest approximated k points.

radius
	

Finds for each element in y all points in x within distance r.

radius_graph
	

Computes graph edges to all points within a given distance.

nearest
	

Finds for each element in y the k nearest point in x.
Unpooling Layers

knn_interpolate
	

The k-NN interpolation from the "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" paper.

Models

MLP
	

A Multi-Layer Perception (MLP) model.

GCN
	

The Graph Neural Network from the "Semi-supervised Classification with Graph Convolutional Networks" paper, using the GCNConv operator for message passing.

GraphSAGE
	

The Graph Neural Network from the "Inductive Representation Learning on Large Graphs" paper, using the SAGEConv operator for message passing.

GIN
	

The Graph Neural Network from the "How Powerful are Graph Neural Networks?" paper, using the GINConv operator for message passing.

GAT
	

The Graph Neural Network from "Graph Attention Networks" or "How Attentive are Graph Attention Networks?" papers, using the GATConv or GATv2Conv operator for message passing, respectively.

PNA
	

The Graph Neural Network from the "Principal Neighbourhood Aggregation for Graph Nets" paper, using the PNAConv operator for message passing.

EdgeCNN
	

The Graph Neural Network from the "Dynamic Graph CNN for Learning on Point Clouds" paper, using the EdgeConv operator for message passing.

JumpingKnowledge
	

The Jumping Knowledge layer aggregation module from the "Representation Learning on Graphs with Jumping Knowledge Networks" paper.

HeteroJumpingKnowledge
	

A heterogeneous version of the JumpingKnowledge module.

MetaLayer
	

A meta layer for building any kind of graph network, inspired by the "Relational Inductive Biases, Deep Learning, and Graph Networks" paper.

Node2Vec
	

The Node2Vec model from the "node2vec: Scalable Feature Learning for Networks" paper where random walks of length walk_length are sampled in a given graph, and node embeddings are learned via negative sampling optimization.

DeepGraphInfomax
	

The Deep Graph Infomax model from the "Deep Graph Infomax" paper based on user-defined encoder and summary model and respectively, and a corruption function .

InnerProductDecoder
	

The inner product decoder from the "Variational Graph Auto-Encoders" paper.

GAE
	

The Graph Auto-Encoder model from the "Variational Graph Auto-Encoders" paper based on user-defined encoder and decoder models.

VGAE
	

The Variational Graph Auto-Encoder model from the "Variational Graph Auto-Encoders" paper.

ARGA
	

The Adversarially Regularized Graph Auto-Encoder model from the "Adversarially Regularized Graph Autoencoder for Graph Embedding" paper.

ARGVA
	

The Adversarially Regularized Variational Graph Auto-Encoder model from the "Adversarially Regularized Graph Autoencoder for Graph Embedding" paper.

SignedGCN
	

The signed graph convolutional network model from the "Signed Graph Convolutional Network" paper.

RENet
	

The Recurrent Event Network model from the "Recurrent Event Network for Reasoning over Temporal Knowledge Graphs" paper.

GraphUNet
	

The Graph U-Net model from the "Graph U-Nets" paper which implements a U-Net like architecture with graph pooling and unpooling operations.

SchNet
	

The continuous-filter convolutional neural network SchNet from the "SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions" paper that uses the interactions blocks of the form.

DimeNet
	

The directional message passing neural network (DimeNet) from the "Directional Message Passing for Molecular Graphs" paper.

DimeNetPlusPlus
	

The DimeNet++ from the "Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules" paper.

GPSE
	

The Graph Positional and Structural Encoder (GPSE) model from the "Graph Positional and Structural Encoder" paper.

GPSENodeEncoder
	

A helper linear/MLP encoder that takes the GPSE encodings (based on the "Graph Positional and Structural Encoder" paper) precomputed as batch.pestat_GPSE in the input graphs, maps them to a desired dimension defined by dim_pe_out and appends them to node features.

to_captum_model
	

Converts a model to a model that can be used for Captum attribution methods.

to_captum_input
	

Given x, edge_index and mask_type, converts it to a format to use in Captum attribution methods.

captum_output_to_dicts
	

Convert the output of Captum attribution methods which is a tuple of attributions to two dictionaries with node and edge attribution tensors.

MetaPath2Vec
	

The MetaPath2Vec model from the "metapath2vec: Scalable Representation Learning for Heterogeneous Networks" paper where random walks based on a given metapath are sampled in a heterogeneous graph, and node embeddings are learned via negative sampling optimization.

DeepGCNLayer
	

The skip connection operations from the "DeepGCNs: Can GCNs Go as Deep as CNNs?" and "All You Need to Train Deeper GCNs" papers.

TGNMemory
	

The Temporal Graph Network (TGN) memory model from the "Temporal Graph Networks for Deep Learning on Dynamic Graphs" paper.

LabelPropagation
	

The label propagation operator, firstly introduced in the "Learning from Labeled and Unlabeled Data with Label Propagation" paper.

CorrectAndSmooth
	

The correct and smooth (C&S) post-processing model from the "Combining Label Propagation And Simple Models Out-performs Graph Neural Networks" paper, where soft predictions (obtained from a simple base predictor) are first corrected based on ground-truth training label information and residual propagation.

AttentiveFP
	

The Attentive FP model for molecular representation learning from the "Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism" paper, based on graph attention mechanisms.

RECT_L
	

The RECT model, i.e. its supervised RECT-L part, from the "Network Embedding with Completely-imbalanced Labels" paper.

LINKX
	

The LINKX model from the "Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods" paper.

LightGCN
	

The LightGCN model from the "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" paper.

MaskLabel
	

The label embedding and masking layer from the "Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification" paper.

GroupAddRev
	

The Grouped Reversible GNN module from the "Graph Neural Networks with 1000 Layers" paper.

GNNFF
	

The Graph Neural Network Force Field (GNNFF) from the "Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture" paper.

PMLP
	

The P(ropagational)MLP model from the "Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs" paper.

NeuralFingerprint
	

The Neural Fingerprint model from the "Convolutional Networks on Graphs for Learning Molecular Fingerprints" paper to generate fingerprints of molecules.

ViSNet
	

A PyTorch module that implements the equivariant vector-scalar interactive graph neural network (ViSNet) from the "Enhancing Geometric Representations for Molecules with Equivariant Vector-Scalar Interactive Message Passing" paper.

LPFormer
	

The LPFormer model from the "LPFormer: An Adaptive Graph Transformer for Link Prediction" paper.

SGFormer
	

The sgformer module from the "SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations" paper.

Polynormer
	

The polynormer module from the "Polynormer: polynomial-expressive graph transformer in linear time" paper.

ARLinkPredictor
	

Link predictor using Attract-Repel embeddings from the paper "Pseudo-Euclidean Attract-Repel Embeddings for Undirected Graphs".
KGE Models

KGEModel
	

An abstract base class for implementing custom KGE models.

TransE
	

The TransE model from the "Translating Embeddings for Modeling Multi-Relational Data" paper.

ComplEx
	

The ComplEx model from the "Complex Embeddings for Simple Link Prediction" paper.

DistMult
	

The DistMult model from the "Embedding Entities and Relations for Learning and Inference in Knowledge Bases" paper.

RotatE
	

The RotatE model from the "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space" paper.
Encodings

PositionalEncoding
	

The positional encoding scheme from the "Attention Is All You Need" paper.

TemporalEncoding
	

The time-encoding function from the "Do We Really Need Complicated Model Architectures for Temporal Networks?" paper.
Functional

bro
	

The Batch Representation Orthogonality penalty from the "Improving Molecular Graph Neural Network Explainability with Orthonormalization and Induced Sparsity" paper.

gini
	

The Gini coefficient from the "Improving Molecular Graph Neural Network Explainability with Orthonormalization and Induced Sparsity" paper.

Dense Convolutional Layers

DenseGCNConv
	

See torch_geometric.nn.conv.GCNConv.

DenseGINConv
	

See torch_geometric.nn.conv.GINConv.

DenseGraphConv
	

See torch_geometric.nn.conv.GraphConv.

DenseSAGEConv
	

See torch_geometric.nn.conv.SAGEConv.

DenseGATConv
	

See torch_geometric.nn.conv.GATConv.
Dense Pooling Layers

dense_diff_pool
	

The differentiable pooling operator from the "Hierarchical Graph Representation Learning with Differentiable Pooling" paper.

dense_mincut_pool
	

The MinCut pooling operator from the "Spectral Clustering in Graph Neural Networks for Graph Pooling" paper.

DMoNPooling
	

The spectral modularity pooling operator from the "Graph Clustering with Graph Neural Networks" paper.
Model Transformations

class Transformer(module: Module, input_map: Optional[Dict[str, str]] = None, debug: bool = False)[source]

    A Transformer executes an FX graph node-by-node, applies transformations to each node, and produces a new torch.nn.Module. It exposes a transform() method that returns the transformed Module. Transformer works entirely symbolically.

    Methods in the Transformer class can be overridden to customize the behavior of transformation.

    transform()
        +-- Iterate over each node in the graph
            +-- placeholder()
            +-- get_attr()
            +-- call_function()
            +-- call_method()
            +-- call_module()
            +-- call_message_passing_module()
            +-- call_global_pooling_module()
            +-- output()
        +-- Erase unused nodes in the graph
        +-- Iterate over each children module
            +-- init_submodule()

    In contrast to the torch.fx.Transformer class, the Transformer exposes additional functionality:

        It subdivides call_module() into nodes that call a regular torch.nn.Module (call_module()), a MessagePassing module (call_message_passing_module()), or a GlobalPooling module (call_global_pooling_module()).

        It allows to customize or initialize new children modules via init_submodule()

        It allows to infer whether a node returns node-level or edge-level information via is_edge_level().

    Parameters:

            module (torch.nn.Module) – The module to be transformed.

            input_map (Dict[str, str], optional) – A dictionary holding information about the type of input arguments of module.forward. For example, in case arg is a node-level argument, then input_map['arg'] = 'node', and input_map['arg'] = 'edge' otherwise. In case input_map is not further specified, will try to automatically determine the correct type of input arguments. (default: None)

            debug (bool, optional) – If set to True, will perform transformation in debug mode. (default: False)

    placeholder(node: Node, target: Any, name: str)[source]

    get_attr(node: Node, target: Any, name: str)[source]

    call_message_passing_module(node: Node, target: Any, name: str)[source]

    call_global_pooling_module(node: Node, target: Any, name: str)[source]

    call_module(node: Node, target: Any, name: str)[source]

    call_method(node: Node, target: Any, name: str)[source]

    call_function(node: Node, target: Any, name: str)[source]

    output(node: Node, target: Any, name: str)[source]

    init_submodule(module: Module, target: str) → Module[source]

        Return type:

            Module

    transform() → GraphModule[source]

        Transforms self.module and returns a transformed torch.fx.GraphModule.

        Return type:

            GraphModule

    is_node_level(node: Node) → bool[source]

        Return type:

            bool

    is_edge_level(node: Node) → bool[source]

        Return type:

            bool

    is_graph_level(node: Node) → bool[source]

        Return type:

            bool

    has_node_level_arg(node: Node) → bool[source]

        Return type:

            bool

    has_edge_level_arg(node: Node) → bool[source]

        Return type:

            bool

    has_graph_level_arg(node: Node) → bool[source]

        Return type:

            bool

    replace_all_uses_with(to_replace: Node, replace_with: Node)[source]

to_hetero(module: Module, metadata: Tuple[List[str], List[Tuple[str, str, str]]], aggr: str = 'sum', input_map: Optional[Dict[str, str]] = None, debug: bool = False) → GraphModule[source]

    Converts a homogeneous GNN model into its heterogeneous equivalent in which node representations are learned for each node type in metadata[0], and messages are exchanged between each edge type in metadata[1], as denoted in the “Modeling Relational Data with Graph Convolutional Networks” paper.

    import torch
    from torch_geometric.nn import SAGEConv, to_hetero

    class GNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), 32)
            self.conv2 = SAGEConv((32, 32), 32)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            return x

    model = GNN()

    node_types = ['paper', 'author']
    edge_types = [
        ('paper', 'cites', 'paper'),
        ('paper', 'written_by', 'author'),
        ('author', 'writes', 'paper'),
    ]
    metadata = (node_types, edge_types)

    model = to_hetero(model, metadata)
    model(x_dict, edge_index_dict)

    where x_dict and edge_index_dict denote dictionaries that hold node features and edge connectivity information for each node type and edge type, respectively.

    The below illustration shows the original computation graph of the homogeneous model on the left, and the newly obtained computation graph of the heterogeneous model on the right:
    ../_images/to_hetero.svg

    Transforming a model via to_hetero().

    Here, each MessagePassing instance
    is duplicated and stored in a set
    (one instance for each relation in ), and message passing in layer is performed via

    where denotes the neighborhood of under relation , and denotes the aggregation scheme aggr to use for grouping node embeddings generated by different relations ("sum", "mean", "min", "max" or "mul").

    Parameters:

            module (torch.nn.Module) – The homogeneous model to transform.

            metadata (Tuple[List[str], List[Tuple[str, str, str]]]) – The metadata of the heterogeneous graph, i.e. its node and edge types given by a list of strings and a list of string triplets, respectively. See HeteroData.metadata() for more information.

            aggr (str, optional) – The aggregation scheme to use for grouping node embeddings generated by different relations ("sum", "mean", "min", "max", "mul"). (default: "sum")

            input_map (Dict[str, str], optional) – A dictionary holding information about the type of input arguments of module.forward. For example, in case arg is a node-level argument, then input_map['arg'] = 'node', and input_map['arg'] = 'edge' otherwise. In case input_map is not further specified, will try to automatically determine the correct type of input arguments. (default: None)

            debug (bool, optional) – If set to True, will perform transformation in debug mode. (default: False)

    Return type:

        GraphModule

to_hetero_with_bases(module: Module, metadata: Tuple[List[str], List[Tuple[str, str, str]]], num_bases: int, in_channels: Optional[Dict[str, int]] = None, input_map: Optional[Dict[str, str]] = None, debug: bool = False) → GraphModule[source]

    Converts a homogeneous GNN model into its heterogeneous equivalent via the basis-decomposition technique introduced in the “Modeling Relational Data with Graph Convolutional Networks” paper.

    For this, the heterogeneous graph is mapped to a typed homogeneous graph, in which its feature representations are aligned and grouped to a single representation. All GNN layers inside the model will then perform message passing via basis-decomposition regularization. This transformation is especially useful in highly multi-relational data, such that the number of parameters no longer depend on the number of relations of the input graph:

    import torch
    from torch_geometric.nn import SAGEConv, to_hetero_with_bases

    class GNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv((16, 16), 32)
            self.conv2 = SAGEConv((32, 32), 32)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            return x

    model = GNN()

    node_types = ['paper', 'author']
    edge_types = [
        ('paper', 'cites', 'paper'),
        ('paper', 'written_by', 'author'),
        ('author', 'writes', 'paper'),
    ]
    metadata = (node_types, edge_types)

    model = to_hetero_with_bases(model, metadata, num_bases=3,
                                 in_channels={'x': 16})
    model(x_dict, edge_index_dict)

    where x_dict and edge_index_dict denote dictionaries that hold node features and edge connectivity information for each node type and edge type, respectively. In case in_channels is given for a specific input argument, its heterogeneous feature information is first aligned to the given dimensionality.

    The below illustration shows the original computation graph of the homogeneous model on the left, and the newly obtained computation graph of the regularized heterogeneous model on the right:
    ../_images/to_hetero_with_bases.svg

    Transforming a model via to_hetero_with_bases().

    Here, each MessagePassing instance
    is duplicated num_bases times and stored in a set
    (one instance for each basis in num_bases), and message passing in layer is performed via

    where denotes the neighborhood of under relation . Notably, only the trainable basis coefficients
    depend on the relations in .

    Parameters:

            module (torch.nn.Module) – The homogeneous model to transform.

            metadata (Tuple[List[str], List[Tuple[str, str, str]]]) – The metadata of the heterogeneous graph, i.e. its node and edge types given by a list of strings and a list of string triplets, respectively. See HeteroData.metadata() for more information.

            num_bases (int) – The number of bases to use.

            in_channels (Dict[str, int], optional) – A dictionary holding information about the desired input feature dimensionality of input arguments of module.forward. In case in_channels is given for a specific input argument, its heterogeneous feature information is first aligned to the given dimensionality. This allows handling of node and edge features with varying feature dimensionality across different types. (default: None)

            input_map (Dict[str, str], optional) – A dictionary holding information about the type of input arguments of module.forward. For example, in case arg is a node-level argument, then input_map['arg'] = 'node', and input_map['arg'] = 'edge' otherwise. In case input_map is not further specified, will try to automatically determine the correct type of input arguments. (default: None)

            debug (bool, optional) – If set to True, will perform transformation in debug mode. (default: False)

    Return type:

        GraphModule

DataParallel Layers

Warning

DataParallel is deprecated. Please use torch.nn.parallel.DistributedDataParallel instead.

class DataParallel(module, device_ids=None, output_device=None, follow_batch=None, exclude_keys=None)[source]

    Implements data parallelism at the module level.

    This container parallelizes the application of the given module by splitting a list of torch_geometric.data.Data objects and copying them as torch_geometric.data.Batch objects to each device. In the forward pass, the module is replicated on each device, and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used.

    The parallelized module must have its parameters and buffers on device_ids[0].

    Note

    You need to use the torch_geometric.loader.DataListLoader for this module.

    Warning

    It is recommended to use torch.nn.parallel.DistributedDataParallel instead of DataParallel for multi-GPU training. DataParallel is usually much slower than DistributedDataParallel even on a single machine. Take a look here for an example on how to use PyG in combination with DistributedDataParallel.

    Parameters:

            module (Module) – Module to be parallelized.

            device_ids (list of int or torch.device) – CUDA devices. (default: all devices)

            output_device (int or torch.device) – Device location of output. (default: device_ids[0])

            follow_batch (list or tuple, optional) – Creates assignment batch vectors for each key in the list. (default: None)

            exclude_keys (list or tuple, optional) – Will exclude each key in the list. (default: None)

Model Hub

class PyGModelHubMixin(model_name: str, dataset_name: str, model_kwargs: Dict)[source]

    A mixin for saving and loading models to the Huggingface Model Hub.

    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import Node2Vec
    from torch_geometric.nn.model_hub import PyGModelHubMixin

    # Define your class with the mixin:
    class N2V(Node2Vec, PyGModelHubMixin):
        def __init__(self,model_name, dataset_name, model_kwargs):
            Node2Vec.__init__(self,**model_kwargs)
            PyGModelHubMixin.__init__(self, model_name,
                dataset_name, model_kwargs)

    # Instantiate your model:
    n2v = N2V(model_name='node2vec',
        dataset_name='Cora', model_kwargs=dict(
        edge_index=data.edge_index, embedding_dim=128,
        walk_length=20, context_size=10, walks_per_node=10,
        num_negative_samples=1, p=1, q=1, sparse=True))

    # Train the model:
    ...

    # Push to the HuggingFace hub:
    repo_id = ...  # your repo id
    n2v.save_pretrained(
        local_file_path,
        push_to_hub=True,
        repo_id=repo_id,
     )

    # Load the model for inference:
    # The required arguments are the repo id/local folder, and any model
    # initialisation arguments that are not native python types (e.g
    # Node2Vec requires the edge_index argument which is not stored in the
    # model hub).
    model = N2V.from_pretrained(
        repo_id,
        model_name='node2vec',
        dataset_name='Cora',
        edge_index=data.edge_index,
    )

    Parameters:

            model_name (str) – Name of the model.

            dataset_name (str) – Name of the dataset the model was trained against.

            model_kwargs (Dict[str, Any]) – The arguments to initialise the model.

    save_pretrained(save_directory: Union[str, Path], push_to_hub: bool = False, repo_id: Optional[str] = None, **kwargs)[source]

        Save a trained model to a local directory or to the HuggingFace model hub.

        Parameters:

                save_directory (str) – The directory where weights are saved.

                push_to_hub (bool, optional) – If True, push the model to the HuggingFace model hub. (default: False)

                repo_id (str, optional) – The repository name in the hub. If not provided will default to the name of save_directory in your namespace. (default: None)

                **kwargs – Additional keyword arguments passed to huggingface_hub.ModelHubMixin.save_pretrained().

    classmethod from_pretrained(pretrained_model_name_or_path: str, force_download: bool = False, token: Optional[Union[str, bool]] = None, cache_dir: Optional[str] = None, local_files_only: bool = False, **model_kwargs) → Any[source]

        Downloads and instantiates a model from the HuggingFace hub.

        Parameters:

                pretrained_model_name_or_path (str) –

                Can be either:

                    The model_id of a pretrained model hosted inside the HuggingFace hub.

                    You can add a revision by appending @ at the end of model_id to load a specific model version.

                    A path to a directory containing the saved model weights.

                    None if you are both providing the configuration config and state dictionary state_dict.

                force_download (bool, optional) – Whether to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist. (default: False)

                token (str or bool, optional) – The token to use as HTTP bearer authorization for remote files. If set to True, will use the token generated when running transformers-cli login (stored in huggingface). It is required if you want to use a private model. (default: None)

                cache_dir (str, optional) – The path to a directory in which a downloaded model configuration should be cached if the standard cache should not be used. (default: None)

                local_files_only (bool, optional) – Whether to only look at local files, i.e. do not try to download the model. (default: False)

                **model_kwargs – Additional keyword arguments passed to the model during initialization.

        Return type:

            Any

Model Summary

summary(model: Module, *args, max_depth: int = 3, leaf_module: Optional[Union[Module, List[Module]]] = 'MessagePassing', **kwargs) → str[source]

    Summarizes a given torch.nn.Module. The summarized information includes (1) layer names, (2) input and output shapes, and (3) the number of parameters.

    import torch
    from torch_geometric.nn import GCN, summary

    model = GCN(128, 64, num_layers=2, out_channels=32)
    x = torch.randn(100, 128)
    edge_index = torch.randint(100, size=(2, 20))

    print(summary(model, x, edge_index))

    +---------------------+---------------------+--------------+--------+
    | Layer               | Input Shape         | Output Shape | #Param |
    |---------------------+---------------------+--------------+--------|
    | GCN                 | [100, 128], [2, 20] | [100, 32]    | 10,336 |
    | ├─(act)ReLU         | [100, 64]           | [100, 64]    | --     |
    | ├─(convs)ModuleList | --                  | --           | 10,336 |
    | │    └─(0)GCNConv   | [100, 128], [2, 20] | [100, 64]    | 8,256  |
    | │    └─(1)GCNConv   | [100, 64], [2, 20]  | [100, 32]    | 2,080  |
    +---------------------+---------------------+--------------+--------+

    Parameters:

            model (torch.nn.Module) – The model to summarize.

            *args – The arguments of the model.

            max_depth (int, optional) – The depth of nested layers to display. Any layers deeper than this depth will not be displayed in the summary. (default: 3)

            leaf_module (torch.nn.Module or [torch.nn.Module], optional) – The modules to be treated as leaf modules, whose submodules are excluded from the summary. (default: MessagePassing)

            **kwargs – Additional arguments of the model.

    Return type:

        str