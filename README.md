### **Categorical Data Clustering**

A Python script for clustering categorical datasets using **DBSCAN** and **HDBSCAN** with different distance metrics. Includes **Jaccard, Overlap, Gower, and Euclidean** distance computations, dimensionality reduction (PCA & t-SNE), and visualization.

### Features
✅ Supports **categorical data clustering**  
✅ Computes **Jaccard, Hamming, Overlap, Gower, and Euclidean** distance matrices  
✅ Uses **DBSCAN** and **HDBSCAN** for clustering  
✅ Provides **t-SNE and PCA visualizations**  
✅ Automatically suggests an **optimal epsilon (eps) value** for DBSCAN  


### Installation
Make sure you have **Python 3.11+** installed. You can install dependencies using [Poetry](https://python-poetry.org/):

```bash
# Install dependencies
poetry install
```

### Usage
Run the script from the terminal with a CSV file containing categorical data:

```bash
python clustering.py data.csv --distance_mode jaccard --cluster_mode dbscan --eps 0.6 --min_samples 50
```

#### Command Line Arguments
| Argument        | Description                                        | Default |
|----------------|----------------------------------------------------|---------|
| `csv_file_path` | Path to the CSV file                              | Required |
| `--re_calc`     | Recompute the distance matrix (`True`/`False`)    | `True` |
| `--distance_mode` | Distance metric (`jaccard`, `hamming`, `overlap`, `gower`, `euclidean`) | `jaccard` |
| `--cluster_mode`  | Clustering method (`dbscan` or `hdbscan`)        | `dbscan` |
| `--eps`         | Epsilon value for DBSCAN clustering               | `0.25` |
| `--min_samples` | MinPts for DBSCAN/HDBSCAN                         | `20` |

### Workflow

1. **Load & Preprocess Data**:  
   - Drops sparse rows/columns  
   - Keeps only categorical features  
2. **Distance Matrix Computation**:  
   - Supports **Jaccard, Overlap, Gower, and Euclidean** distances  
3. **Clustering**:  
   - **DBSCAN** (Density-Based)  
   - **HDBSCAN** (Hierarchical Density-Based)  
4. **Dimensionality Reduction**:  
   - **PCA** for 2D visualization  
   - **t-SNE** for high-dimensional embedding  
5. **Cluster Visualization**:  
   - **2D & 3D scatter plots**  

### Hyperparameter Selection Guide
Choosing the right **eps**, **min_samples**, and **distance metric** is crucial for effective clustering. Use the following table to guide your choices:

| **Parameter**     | **Description** | **Recommended Value** |
|-------------------|----------------|----------------------|
| `eps` (DBSCAN)   | Maximum distance between points in the same cluster. Use the **Elbow Method** to determine this. | Use `opt_eps()` function (default ~0.6-0.75) |
| `min_samples`    | Minimum points to form a dense region. Higher values reduce noise but may miss small clusters. | `2 * num_features` (start with `20-50`) |
| `distance_mode`  | Defines how similarity is measured. | - **Jaccard**: Best for binary or one-hot encoded categorical data.<br> - **Gower**: Handles mixed categorical + numerical data.<br> - **Euclidean**: Only for one-hot encoded data.<br> - **Overlap**: Simple categorical comparison. |
| `cluster_mode`   | Determines which clustering algorithm to use. | - **DBSCAN**: Best for dense clusters.<br> - **HDBSCAN**: Better for variable density clusters. |
| `t-SNE perplexity` | Controls how t-SNE balances local vs. global structure. | 5-50 (default: `30`) |

### Combinations

> Most clustering algorithms, especially those that are distance-based like **DBSCAN** and **HDBSCAN**, rely on a measure of <u>dissimilarity (or distance)</u> between data points.

> **One-hot encoded data alone**: Jaccard Distance is typically the better and more common choice. Overlap Coefficient can be used but is less common and often less effective.

> **One-hot encoded data mixed with numerical data**: Gower Distance is a very good choice as it handles both categorical and numerical features effectively.

| **Distance/Dissimilarity Metric** | **Description**                                                                     | **DBSCAN**                               | **HDBSCAN**                               |
|-----------------------------------|---------------------------------------------------------------------------------|------------------------------------------|-------------------------------------------|
| **Jaccard Distance**              | Measures dissimilarity based on set intersection and union (ideal for binary data). | Yes, works well for one-hot encoded data. | Yes, works well for one-hot encoded data.  |
| **Overlap Coefficient**           | Measures similarity based on the intersection of features (binary vectors). Less common than Jaccard for measuring dissimilarity. | Yes, requires custom *dissimilarity* function (e.g., 1 - Overlap Coefficient). | Can be used with customization, but less common than Jaccard. |
| **Gower Distance**                | Works for mixed data types (both numeric and categorical features).                 | Yes, with custom metric support (best for mixed data). | Yes, works well with mixed data.          |
| **Euclidean Distance**            | Measures straight-line distance between points (suitable for continuous data).     | Not recommended for one-hot encoded categorical data. | Not recommended for one-hot encoded categorical data. |
| **Hamming Distance**             | Counts the number of positions at which two vectors are different. Well-suited for binary/one-hot encoded data. | Yes, works well for one-hot encoded data. | Yes, works well for one-hot encoded data. |

### Reference

- https://www.datacamp.com/tutorial/dbscan-clustering-algorithm
- https://www.freecodecamp.org/news/clustering-in-python-a-machine-learning-handbook/