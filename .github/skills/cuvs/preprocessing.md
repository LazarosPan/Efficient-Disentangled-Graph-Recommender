Preprocessing
Binary Quantizer

cuvs.preprocessing.quantize.binary.transform(dataset, output=None, resources=None)
[source]

    Applies binary quantization transform to given dataset

    This applies binary quantization to a dataset, changing any positive values to a bitwise 1. This is useful for searching with the BitwiseHamming distance type.

    Parameters:

        dataset
        row major host or device dataset to transform
        output
        optional preallocated output memory, on host or device memory
        resources
        Optional cuVS Resource handle for reusing CUDA resources.

            If Resources aren’t supplied, CUDA resources will be allocated inside this function and synchronized before the function exits. If resources are supplied, you will need to explicitly synchronize yourself by calling resources.sync() before accessing the output.

    Returns:

        output
        transformed dataset quantized into a uint8

    Examples

    import cupy as cp
    from cuvs.preprocessing.quantize import binary
    from cuvs.neighbors import cagra
    n_samples = 50000
    n_features = 50
    dataset = cp.random.standard_normal((n_samples, n_features),
                                      dtype=cp.float32)
    transformed = binary.transform(dataset)

    # build a cagra index on the binarized data
    params = cagra.IndexParams(metric="bitwise_hamming",
                               build_algo="iterative_cagra_search")
    idx = cagra.build(params, transformed)

Product Quantizer

class cuvs.preprocessing.quantize.pq.Quantizer

    Defines and stores Product Quantizer upon training

    The quantization is performed by a linear mapping of an interval in the float data type to the full range of the quantized int type.

    Attributes:

        encoded_dim

            Returns the encoded dimension of the quantized dataset
        pq_bits
        pq_codebook

            Returns the PQ codebook
        pq_dim
        use_vq
        vq_codebook

            Returns the VQ codebook

    encoded_dim

        Returns the encoded dimension of the quantized dataset

    pq_codebook

        Returns the PQ codebook

    vq_codebook

        Returns the VQ codebook

class cuvs.preprocessing.quantize.pq.QuantizerParams(

    pq_bits=8,
    *,
    pq_dim=0,
    use_subspaces=True,
    use_vq=False,
    vq_n_centers=0,
    kmeans_n_iters=25,
    pq_kmeans_type='kmeans_balanced',
    max_train_points_per_pq_code=256,
    max_train_points_per_vq_cluster=1024,

)

    Parameters for product quantization

    Parameters:

        pq_bits: int

            specifies the bit length of the vector element after compression by PQ possible values: within [4, 16]
        pq_dim: int

            specifies the dimensionality of the vector after compression by PQ
        use_subspaces: bool

            specifies whether to use subspaces for product quantization (PQ). When true, one PQ codebook is used for each subspace. Otherwise, a single PQ codebook is used.
        use_vq: bool

            specifies whether to use Vector Quantization (KMeans) before product quantization (PQ).
        vq_n_centers: int

            specifies the number of centers for the vector quantizer. When zero, an optimal value is selected using a heuristic. When one, only product quantization is used.
        kmeans_n_iters: int

            specifies the number of iterations searching for kmeans centers
        pq_kmeans_type: str

            specifies the type of kmeans algorithm to use for PQ training possible values: “kmeans”, “kmeans_balanced”
        max_train_points_per_pq_code: int

            specifies the max number of data points to use per PQ code during PQ codebook training. Using more data points per PQ code may increase the quality of PQ codebook but may also increase the build time.
        max_train_points_per_vq_cluster: int

            specifies the max number of data points to use per VQ cluster.

    Attributes:

        kmeans_n_iters
        max_train_points_per_pq_code
        max_train_points_per_vq_cluster
        pq_bits
        pq_dim
        pq_kmeans_type
        use_subspaces
        use_vq
        vq_n_centers

cuvs.preprocessing.quantize.pq.build(QuantizerParams params, dataset, resources=None)
[source]

    Builds a Product Quantizer to be used later for quantizing the dataset.

    Parameters:

        params
        QuantizerParams object
        dataset
        row major dataset on host or device memory. FP32
        resources
        Optional cuVS Resource handle for reusing CUDA resources.

            If Resources aren’t supplied, CUDA resources will be allocated inside this function and synchronized before the function exits. If resources are supplied, you will need to explicitly synchronize yourself by calling resources.sync() before accessing the output.

    Returns:

        quantizer: cuvs.preprocessing.quantize.pq.Quantizer

    Examples

    import cupy as cp
    from cuvs.preprocessing.quantize import pq
    n_samples = 5000
    n_features = 64
    dataset = cp.random.random_sample((n_samples, n_features),
                                      dtype=cp.float32)
    params = pq.QuantizerParams(pq_bits=8, pq_dim=16)
    quantizer = pq.build(params, dataset)
    transformed, _ = pq.transform(quantizer, dataset)

cuvs.preprocessing.quantize.pq.transform(

    Quantizer quantizer,
    dataset,
    codes_output=None,
    vq_labels=None,
    resources=None,

)
[source]

    Applies Product Quantization transform to given dataset

    Parameters:

        quantizer
        trained Quantizer object
        dataset
        row major dataset on host or device memory. FP32
        codes_output
        optional preallocated output memory, on device memory
        vq_labels
        optional preallocated output memory for VQ labels, on device memory
        resources
        Optional cuVS Resource handle for reusing CUDA resources.

            If Resources aren’t supplied, CUDA resources will be allocated inside this function and synchronized before the function exits. If resources are supplied, you will need to explicitly synchronize yourself by calling resources.sync() before accessing the output.

    Returns:

        codes_output
        transformed dataset quantized into a uint8
        vq_labels
        VQ labels when VQ is used, None otherwise

    Examples

    import cupy as cp
    from cuvs.preprocessing.quantize import pq
    n_samples = 5000
    n_features = 64
    dataset = cp.random.random_sample((n_samples, n_features),
                                      dtype=cp.float32)
    params = pq.QuantizerParams(pq_bits=8, pq_dim=16)
    quantizer = pq.build(params, dataset)
    transformed, _ = pq.transform(quantizer, dataset)

cuvs.preprocessing.quantize.pq.inverse_transform(

    Quantizer quantizer,
    codes,
    output=None,
    vq_labels=None,
    resources=None,

)
[source]

    Applies Product Quantization inverse transform to given codes

    Parameters:

        quantizer
        trained Quantizer object
        codes
        row major device codes to inverse transform. uint8
        output
        optional preallocated output memory, on device memory
        vq_labels
        optional VQ labels when VQ is used, on device memory
        resources
        Optional cuVS Resource handle for reusing CUDA resources.

            If Resources aren’t supplied, CUDA resources will be allocated inside this function and synchronized before the function exits. If resources are supplied, you will need to explicitly synchronize yourself by calling resources.sync() before accessing the output.

    Returns:

        output
        Original dataset reconstructed from quantized codes

    Examples

    import cupy as cp
    from cuvs.preprocessing.quantize import pq
    n_samples = 5000
    n_features = 64
    dataset = cp.random.random_sample((n_samples, n_features),
                                      dtype=cp.float32)
    params = pq.QuantizerParams(pq_bits=8, pq_dim=16, use_vq=True)
    quantizer = pq.build(params, dataset)
    transformed, vq_labels = pq.transform(quantizer, dataset)
    reconstructed = pq.inverse_transform(quantizer, transformed, vq_labels=vq_labels)

Scalar Quantizer

class cuvs.preprocessing.quantize.scalar.Quantizer

    Defines and stores scalar for quantisation upon training

    The quantization is performed by a linear mapping of an interval in the float data type to the full range of the quantized int type.

    Attributes:

        max
        min

class cuvs.preprocessing.quantize.scalar.QuantizerParams(quantile=None, *)

    Parameters for scalar quantization

    Parameters:

        quantile: float

            specifies how many outliers at top & bottom will be ignored needs to be within range of (0, 1]

    Attributes:

        quantile

cuvs.preprocessing.quantize.scalar.train(QuantizerParams params, dataset, resources=None)
[source]

    Initializes a scalar quantizer to be used later for quantizing the dataset.

    Parameters:

        params
        QuantizerParams object
        dataset
        row major host or device dataset
        resources
        Optional cuVS Resource handle for reusing CUDA resources.

            If Resources aren’t supplied, CUDA resources will be allocated inside this function and synchronized before the function exits. If resources are supplied, you will need to explicitly synchronize yourself by calling resources.sync() before accessing the output.

    Returns:

        quantizer: cuvs.preprocessing.quantize.scalar.Quantizer

    Examples

    import cupy as cp
    from cuvs.preprocessing.quantize import scalar
    n_samples = 50000
    n_features = 50
    dataset = cp.random.random_sample((n_samples, n_features),
                                      dtype=cp.float32)
    params = scalar.QuantizerParams(quantile=0.99)
    quantizer = scalar.train(params, dataset)
    transformed = scalar.transform(quantizer, dataset)

cuvs.preprocessing.quantize.scalar.transform(Quantizer quantizer, dataset, output=None, resources=None)
[source]

    Applies quantization transform to given dataset

    Parameters:

        quantizer
        trained Quantizer object
        dataset
        row major host or device dataset to transform
        output
        optional preallocated output memory, on host or device memory
        resources
        Optional cuVS Resource handle for reusing CUDA resources.

            If Resources aren’t supplied, CUDA resources will be allocated inside this function and synchronized before the function exits. If resources are supplied, you will need to explicitly synchronize yourself by calling resources.sync() before accessing the output.

    Returns:

        output
        transformed dataset quantized into a int8

    Examples

    import cupy as cp
    from cuvs.preprocessing.quantize import scalar
    n_samples = 50000
    n_features = 50
    dataset = cp.random.random_sample((n_samples, n_features),
                                      dtype=cp.float32)
    params = scalar.QuantizerParams(quantile=0.99)
    quantizer = scalar.train(params, dataset)
    transformed = scalar.transform(quantizer, dataset)

cuvs.preprocessing.quantize.scalar.inverse_transform(

    Quantizer quantizer,
    dataset,
    output=None,
    resources=None,

)
[source]

    Perform inverse quantization step on previously quantized dataset

    Note that depending on the chosen data types train dataset the conversion is not lossless.

    Parameters:

        quantizer
        trained Quantizer object
        dataset
        row major host or device dataset to transform
        output
        optional preallocated output memory, on host or device
        resources
        Optional cuVS Resource handle for reusing CUDA resources.

            If Resources aren’t supplied, CUDA resources will be allocated inside this function and synchronized before the function exits. If resources are supplied, you will need to explicitly synchronize yourself by calling resources.sync() before accessing the output.

    Returns:

        output
        transformed dataset with scalar quantization reversed