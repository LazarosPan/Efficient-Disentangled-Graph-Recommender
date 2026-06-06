# torch_geometric.utils

Graph neural network utilities and tools for tensor operations on graphs.

| Function | Description |
|----------|-------------|
| `scatter` | Reduces all values from the src tensor at the indices specified in the index tensor along a given dimension dim. |
| `group_argsort` | Returns the indices that sort the tensor src along a given dimension in ascending order by value. |
| `group_cat` | Concatenates the given sequence of tensors tensors in the given dimension dim. |
| `segment` | Reduces all values in the first dimension of the src tensor within the ranges specified in the ptr. |
| `segment_logsumexp` | Returns the log summed exponentials of each row of the src tensor within the ranges specified in the ptr. |
| `index_sort` | Sorts the elements of the inputs tensor in ascending order. |
| `cumsum` | Returns the cumulative sum of elements of x. |
| `degree` | Computes the (unweighted) degree of a given one-dimensional index tensor. |
| `softmax` | Computes a sparsely evaluated softmax. |
| `lexsort` | Performs an indirect stable sort using a sequence of keys. |
| `sort_edge_index` | Row-wise sorts edge_index. |
| `coalesce` | Row-wise sorts edge_index and removes its duplicated entries. |
| `is_undirected` | Returns True if the graph given by edge_index is undirected. |
| `to_undirected` | Converts the graph given by edge_index to an undirected graph such that $(j,i) \in \mathcal{E}$ for every edge $(i,j) \in \mathcal{E}$. |
| `contains_self_loops` | Returns True if the graph given by edge_index contains self-loops. |
| `remove_self_loops` | Removes every self-loop in the graph given by edge_index, so that $(i,i) \not\in \mathcal{E}$ for every $i \in \mathcal{V}$. |
| `segregate_self_loops` | Segregates self-loops from the graph. |
| `add_self_loops` | Adds a self-loop $(i,i) \in \mathcal{E}$ to every node $i \in \mathcal{V}$ in the graph given by edge_index. |
| `add_remaining_self_loops` | Adds remaining self-loop $(i,i) \in \mathcal{E}$ to every node $i \in \mathcal{V}$ in the graph given by edge_index. |
| `get_self_loop_attr` | Returns the edge features or weights of self-loops $(i, i)$ of every node $i \in \mathcal{V}$ in the graph given by edge_index. |
| `contains_isolated_nodes` | Returns True if the graph given by edge_index contains isolated nodes. |
| `remove_isolated_nodes` | Removes the isolated nodes from the graph given by edge_index with optional edge attributes edge_attr. |
| `get_num_hops` | Returns the number of hops the model is aggregating information from. |
| `subgraph` | Returns the induced subgraph of (edge_index, edge_attr) containing the nodes in subset. |
| `bipartite_subgraph` | Returns the induced subgraph of the bipartite graph (edge_index, edge_attr) containing the nodes in subset. |
| `k_hop_subgraph` | Computes the induced subgraph of edge_index around all nodes in node_idx reachable within hops. |
| `dropout_node` | Randomly drops nodes from the adjacency matrix edge_index with probability p using samples from a Bernoulli distribution. |
| `dropout_edge` | Randomly drops edges from the adjacency matrix edge_index with probability p using samples from a Bernoulli distribution. |
| `dropout_path` | Drops edges from the adjacency matrix edge_index based on random walks. |
| `dropout_adj` | Randomly drops edges from the adjacency matrix (edge_index, edge_attr) with probability p using samples from a Bernoulli distribution. |
| `homophily` | The homophily of a graph characterizes how likely nodes with the same label are near each other in a graph. |
| `assortativity` | The degree assortativity coefficient from the "Mixing patterns in networks" paper. |
| `normalize_edge_index` | Applies normalization to the edges of a graph. |
| `get_laplacian` | Computes the graph Laplacian of the graph given by edge_index and optional edge_weight. |
| `get_mesh_laplacian` | Computes the mesh Laplacian of a mesh given by pos and face. |
| `mask_select` | Returns a new tensor which masks the src tensor along the dimension dim according to the boolean mask mask. |
| `index_to_mask` | Converts indices to a mask representation. |
| `mask_to_index` | Converts a mask to an index representation. |
| `select` | Selects the input tensor or input list according to a given index or mask vector. |
| `narrow` | Narrows the input tensor or input list to the specified range. |
| `to_dense_batch` | Given a sparse batch of node features $\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}$ (with indicating the number of nodes in graph), creates a dense node feature tensor $\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}$ (with $N_{\max} = \max_i^B N_i$). |
| `to_dense_adj` | Converts batched sparse adjacency matrices given by edge indices and edge attributes to a single dense batched adjacency matrix. |
| `to_nested_tensor` | Given a contiguous batch of tensors $\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}$ (with indicating the number of elements in example i), creates a nested PyTorch tensor. |
| `from_nested_tensor` | Given a nested PyTorch tensor, creates a contiguous batch of tensors $\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}$, and optionally a batch vector which assigns each element to a specific example. |
| `dense_to_sparse` | Converts a dense adjacency matrix to a sparse adjacency matrix defined by edge indices and edge attributes. |
| `is_torch_sparse_tensor` | Returns True if the input src is a torch.sparse.Tensor (in any sparse layout). |
| `is_sparse` | Returns True if the input src is of type torch.sparse.Tensor (in any sparse layout) or of type torch_sparse.SparseTensor. |
| `to_torch_coo_tensor` | Converts a sparse adjacency matrix defined by edge indices and edge attributes to a torch.sparse.Tensor with layout torch.sparse_coo. |
| `to_torch_csr_tensor` | Converts a sparse adjacency matrix defined by edge indices and edge attributes to a torch.sparse.Tensor with layout torch.sparse_csr. |
| `to_torch_csc_tensor` | Converts a sparse adjacency matrix defined by edge indices and edge attributes to a torch.sparse.Tensor with layout torch.sparse_csc. |
| `to_torch_sparse_tensor` | Converts a sparse adjacency matrix defined by edge indices and edge attributes to a torch.sparse.Tensor with custom layout. |
| `to_edge_index` | Converts a torch.sparse.Tensor or a torch_sparse.SparseTensor to edge indices and edge attributes. |
| `spmm` | Matrix product of sparse matrix with dense matrix. |
| `unbatch` | Splits src according to a batch vector along dimension dim. |
| `unbatch_edge_index` | Splits the edge_index according to a batch vector. |
| `one_hot` | Taskes a one-dimensional index tensor and returns a one-hot encoded representation of it with shape [*, num_classes] that has zeros everywhere except where the index of last dimension matches the corresponding value of the input tensor, in which case it will be 1. |
| `normalized_cut` | Computes the normalized cut $\mathbf{e}_{i,j} \cdot \left( \frac{1}{\deg(i)} + \frac{1}{\deg(j)} \right)$ of a weighted graph given by edge indices and edge attributes. |
| `grid` | Returns the edge indices of a two-dimensional grid graph with height height and width width and its node positions. |
| `geodesic_distance` | Computes (normalized) geodesic distances of a mesh given by pos and face. |
| `to_scipy_sparse_matrix` | Converts a graph given by edge indices and edge attributes to a scipy sparse matrix. |
| `from_scipy_sparse_matrix` | Converts a scipy sparse matrix to edge indices and edge attributes. |
| `to_networkx` | Converts a torch_geometric.data.Data instance to a networkx.Graph if to_undirected is set to True, or a directed networkx.DiGraph otherwise. |
| `from_networkx` | Converts a networkx.Graph or networkx.DiGraph to a torch_geometric.data.Data instance. |
| `to_networkit` | Converts a (edge_index, edge_weight) tuple to a networkit.Graph. |
| `from_networkit` | Converts a networkit.Graph to a (edge_index, edge_weight) tuple. |
| `to_trimesh` | Converts a torch_geometric.data.Data instance to a trimesh.Trimesh. |
| `from_trimesh` | Converts a trimesh.Trimesh to a torch_geometric.data.Data instance. |
| `to_cugraph` | Converts a graph given by edge_index and optional edge_weight into a cugraph graph object. |
| `from_cugraph` | Converts a cugraph graph object into edge_index and optional edge_weight tensors. |
| `to_dgl` | Converts a torch_geometric.data.Data or torch_geometric.data.HeteroData instance to a dgl graph object. |
| `from_dgl` | Converts a dgl graph object to a torch_geometric.data.Data or torch_geometric.data.HeteroData instance. |
| `from_rdmol` | Converts a rdkit.Chem.Mol instance to a torch_geometric.data.Data instance. |
| `to_rdmol` | Converts a torch_geometric.data.Data instance to a rdkit.Chem.Mol instance. |
| `from_smiles` | Converts a SMILES string to a torch_geometric.data.Data instance. |
| `to_smiles` | Converts a torch_geometric.data.Data instance to a SMILES string. |
| `erdos_renyi_graph` | Returns the edge_index of a random Erdos-Renyi graph. |
| `stochastic_blockmodel_graph` | Returns the edge_index of a stochastic blockmodel graph. |
| `barabasi_albert_graph` | Returns the edge_index of a Barabasi-Albert preferential attachment model, where a graph of num_nodes nodes grows by attaching new nodes with num_edges edges that are preferentially attached to existing nodes with high degree. |
| `negative_sampling` | Samples random negative edges of a graph given by edge_index. |
| `batched_negative_sampling` | Samples random negative edges of multiple graphs given by edge_index and batch. |
| `structured_negative_sampling` | Samples a negative edge (i,k) for every positive edge (i,j) in the graph given by edge_index, and returns it as a tuple of the form (i,j,k). |
| `shuffle_node` | Randomly shuffle the feature matrix x along the first dimension. |
| `mask_feature` | Randomly masks feature from the feature matrix x with probability p using samples from a Bernoulli distribution. |
| `add_random_edge` | Randomly adds edges to edge_index. |
| `tree_decomposition` | The tree decomposition algorithm of molecules from the "Junction Tree Variational Autoencoder for Molecular Graph Generation" paper. |
| `get_embeddings` | Returns the output embeddings of all MessagePassing layers in model. |
| `get_embeddings_hetero` | Returns the output embeddings of all MessagePassing layers in a heterogeneous model, organized by edge type. |
| `trim_to_layer` | Trims the edge_index representation, node features x and edge features edge_attr to a minimal-sized representation for the current GNN layer layer in directed NeighborLoader scenarios. |
| `get_ppr` | Calculates the personalized PageRank (PPR) vector for all or a subset of nodes using a variant of the Andersen algorithm. |
| `train_test_split_edges` | Splits the edges of a torch_geometric.data.Data object into positive and negative train/val/test edges. |
| `total_influence` | Compute Jacobian‑based influence aggregates for multiple seed nodes, as introduced in the "Towards Quantifying Long-Range Interactions in Graph Machine Learning: a Large Graph Dataset and a Measurement" paper. |
