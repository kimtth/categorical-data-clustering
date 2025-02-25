import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plotting is available
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import OneHotEncoder


# Function to convert a row to a set of key:value pairs (as tuples)
def row_to_set(row):
    excluded_cols = [
        # Exclude columns that are not used for comparison
        "ID"
    ]
    return {
        (col, str(row[col]).strip().lower())
        for col in row.index
        if col not in excluded_cols
        and pd.notna(row[col])
        and str(row[col]).strip() != ""
    }


def filtered_df_overlap(df):
    # Drop columns that are not used for comparison.
    excluded_cols = ["ID"]
    return df.drop(excluded_cols, axis=1, errors="ignore")


def filter_categorical_columns(df):
    # Select columns with dtype object (usually categorical)
    categorical_df = df.select_dtypes(include=["object"])
    return categorical_df


# Overlap distance computation (element-wise comparison of string arrays)
def overlap_distance(df):
    return squareform(pdist(df, metric=lambda a, b: 1 - (np.sum(a == b) / len(a))))


# Gower Distance
def gower_distance(df):
    # Convert all entries to string so that comparisons are consistent.
    df_str = df.astype(str)
    # Use a lambda function with pdist: for two rows u and v, count mismatches divided by number of columns.
    dist_matrix = squareform(pdist(df_str, metric=lambda u, v: np.mean(u != v)))
    return dist_matrix


def one_hot_encode(df):
    df = df.astype(str)  # Convert all data to strings
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(df)
    print("One-hot encoding completed.")
    return X_encoded


def load_and_prepare_data(csv_file, sample_size=100):
    df = pd.read_csv(csv_file)
    print("Original shape:", df.shape)
    na_threshold_rows = (
        0.4  # For rows, 0.4 means only keep rows with 40% or more non-null values
    )
    na_threshold_cols = (
        0.7  # For columns, 0.2 means only keep columns with 70% or more non-null values
    )
    print(
        f"Threshold for dropping sparse rows and columns: {na_threshold_rows * len(df.columns)} : {na_threshold_cols * len(df)}"
    )
    df = df.dropna(thresh=na_threshold_rows * len(df.columns))
    df = df.dropna(thresh=na_threshold_cols * len(df), axis=1)
    print("After dropping sparse rows and columns:", df.shape)
    df = df.fillna("")  # Fill missing values with empty strings
    df = df.reset_index(drop=True)
    # Keep only categorical columns
    df = filter_categorical_columns(df)
    print("After Filtering categorical columns:", df.shape)
    # Save a sample for exploration.
    df_sample = df.head(sample_size)
    df_sample_clean = filtered_df_overlap(df_sample)
    df_sample_clean.to_csv("df_sample.csv", encoding="utf8", index=False)
    return df


def compute_distance_matrix(df, distance_mode="jaccard", recalc=True):
    if recalc:
        print("Computing distance matrix...")
        df_encoded = one_hot_encode(filtered_df_overlap(df))
        if distance_mode == "jaccard":
            dist_matrix = pairwise_distances(df_encoded, metric="jaccard")
        elif distance_mode == "hamming":
            dist_matrix = pairwise_distances(df_encoded, metric="hamming")
        elif distance_mode == "overlap":
            dist_matrix = overlap_distance(df_encoded)
        elif distance_mode == "gower":
            dist_matrix = gower_distance(df_encoded)
        elif distance_mode == "euclidean":
            dist_matrix = pairwise_distances(df_encoded, metric="euclidean")
        else:
            raise ValueError(f"Unsupported distance mode: {distance_mode}")
        pickle.dump(dist_matrix, open(f"{distance_mode}_dist_matrix.pkl", "wb"))
    else:
        dist_matrix = pickle.load(open(f"{distance_mode}_dist_matrix.pkl", "rb"))
    print("Distance Matrix created.")
    return dist_matrix


def opt_eps(dist_matrix, min_samples):
    neighbors = NearestNeighbors(n_neighbors=min_samples, metric="precomputed")
    neighbors_fit = neighbors.fit(dist_matrix)
    distances, _ = neighbors_fit.kneighbors(dist_matrix)
    distances = np.sort(distances[:, -1])
    plt.plot(distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{min_samples}th Nearest Neighbor Distance")
    plt.title("Elbow Method for Optimal eps")
    plt.show(block=True)
    print("Optimal eps estimation completed.")

    # Automatically suggest an eps value:
    try:
        # Find the "elbow" point (you might need to adjust this logic)
        knee_index = np.argmax(
            np.diff(distances)
        )  # Find the point with the largest increase in distance
        suggested_eps = distances[knee_index]
        print(f"Suggested eps: {suggested_eps}")
        return suggested_eps  # Return suggested eps
    except Exception as e:
        print(f"Could not automatically estimate eps: {e}")
        return None  # Or return a default value if you prefer


def perform_dbscan_clustering(df, dist_matrix, eps=0.5, min_samples=4):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    df["cluster"] = dbscan.fit_predict(dist_matrix)
    clusters = df["cluster"]
    # Print number of clusters and noise points
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    print("DBSCAN clustering completed.")
    return df


def perform_hdbscan_clustering(df, dist_matrix, min_cluster_size=5, min_samples=4):
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed", min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    df["cluster"] = clusterer.fit_predict(dist_matrix)
    cluster_num = len(df["cluster"].unique())
    print(f"Number of clusters: {cluster_num}")
    print("HDBSCAN clustering completed.")
    return df


def perform_tsne_embedding(df, dist_matrix):
    # 2D t-SNE embedding
    tsne_2d = TSNE(
        n_components=2,
        max_iter=1000,
        perplexity=30,
        learning_rate="auto",
        metric="precomputed",
        random_state=42,
        init="random",
    )
    tsne_results_2d = tsne_2d.fit_transform(dist_matrix)
    df["TSNE2D_1"] = tsne_results_2d[:, 0]
    df["TSNE2D_2"] = tsne_results_2d[:, 1]

    # 3D t-SNE embedding
    tsne_3d = TSNE(
        n_components=3,
        max_iter=1000,
        perplexity=30,
        learning_rate="auto",
        metric="precomputed",
        random_state=42,
        init="random",
    )
    tsne_results_3d = tsne_3d.fit_transform(dist_matrix)
    df["TSNE3D_1"] = tsne_results_3d[:, 0]
    df["TSNE3D_2"] = tsne_results_3d[:, 1]
    df["TSNE3D_3"] = tsne_results_3d[:, 2]
    print("t-SNE embedding completed.")
    return df


def plot_clusters(df, dist_matrix):
    clusters = df["cluster"].unique()
    palette = sns.color_palette("deep", len(clusters))
    color_dict = dict(zip(clusters, palette))

    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(131)
    # 1. DBSCAN Cluster Plot (Raw Features)
    # Apply PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    dist_matrix_2d = pca.fit_transform(
        dist_matrix
    )  # Transform the distance matrix into 2D
    for cluster in clusters:
        subset = df[df["cluster"] == cluster]
        ax1.scatter(
            dist_matrix_2d[:, 0],
            dist_matrix_2d[:, 1],
            label=f"Cluster {cluster}",
            s=50,
            color=color_dict[cluster],
        )
    ax1.set_title("Clustering Results (PCA)")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.legend(title="Cluster")

    # 2D Plot
    ax2 = fig.add_subplot(132)
    for cluster in clusters:
        subset = df[df["cluster"] == cluster]
        ax2.scatter(
            subset["TSNE2D_1"],
            subset["TSNE2D_2"],
            label=f"Cluster {cluster}",
            s=50,
            color=color_dict[cluster],
        )
    ax2.set_title("2D Clusters (T-SNE)")
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")
    ax2.legend(title="Cluster")

    # 3D Plot
    ax3 = fig.add_subplot(133, projection="3d")
    for cluster in clusters:
        subset = df[df["cluster"] == cluster]
        ax3.scatter(
            subset["TSNE3D_1"],
            subset["TSNE3D_2"],
            subset["TSNE3D_3"],
            label=f"Cluster {cluster}",
            s=50,
            color=color_dict[cluster],
        )
    ax3.set_title("3D Clusters (T-SNE)")
    ax3.set_xlabel("Component 1")
    ax3.set_ylabel("Component 2")
    ax3.set_zlabel("Component 3")
    ax3.legend(title="Cluster")

    plt.tight_layout()
    print("Cluster plots displayed.")
    # Keep the plot window open until you manually close it
    plt.show(block=True)


def main(
    csv_file_path,
    re_calc=True,
    distance_mode="jaccard",
    cluster_mode="dbscan",
    eps=0.6,
    min_samples=150,
):
    print(
        f"Clustering and visualization started. {cluster_mode} clustering with {distance_mode} distance."
    )
    csv_file = csv_file_path
    df = load_and_prepare_data(csv_file)
    dist_matrix = compute_distance_matrix(df, distance_mode, recalc=re_calc)
    eps = opt_eps(dist_matrix, min_samples)
    if eps is None or np.isinf(eps) or eps <= 0:
        eps = 0.75  # Default value if automatic detection fails.
        print(f"Using default eps value, {eps}")

    # Ensure non-negative distances
    dist_matrix = np.maximum(dist_matrix, 0)

    if cluster_mode.lower() == "dbscan":
        df = perform_dbscan_clustering(
            df, dist_matrix, eps=eps, min_samples=min_samples
        )
    elif cluster_mode.lower() == "hdbscan":
        df = perform_hdbscan_clustering(
            df, dist_matrix, min_cluster_size=min_samples, min_samples=min_samples
        )
    else:
        print("Unsupported clustering mode.")
        return

    df = perform_tsne_embedding(df, dist_matrix)
    plot_clusters(df, dist_matrix)
    print("Clustering and visualization completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorical Data Clustering")
    parser.add_argument("csv_file_path", type=str, help="Path to the CSV file")
    parser.add_argument(
        "--re_calc", type=bool, default=True, help="Recompute the distance matrix"
    )
    parser.add_argument(
        "--distance_mode",
        type=str,
        default="euclidean",
        choices=["jaccard", "overlap", "gower", "euclidean"],
        help="Distance mode",
    )
    parser.add_argument(
        "--cluster_mode",
        type=str,
        default="dbscan",
        choices=["dbscan", "hdbscan"],
        help="Clustering mode",
    )
    parser.add_argument("--eps", type=float, default=0.25, help="Epsilon for DBSCAN")
    parser.add_argument("--min_samples", type=int, default=20, help="MinPts for DBSCAN")

    args = parser.parse_args()

    main(
        csv_file_path=args.csv_file_path,
        re_calc=(
            args.re_calc if args.re_calc is not None else True
        ),  # Set to True if you want to recompute the distance matrix
        distance_mode=(
            args.distance_mode if args.distance_mode is not None else "euclidean"
        ),  # Options: jaccard, overlap, gower, euclidean
        cluster_mode=(
            args.cluster_mode if args.cluster_mode is not None else "dbscan"
        ),  # Options: dbscan, hdbscan
        eps=(
            args.eps if args.eps is not None else 0.25
        ),  # Epsilon for DBSCAN: K-nearest neighbors distance N
        min_samples=(
            args.min_samples if args.min_samples is not None else 20
        ),  # MinPts for DBSCAN: A good starting point is to set MinPts = 2 * num_features
    )
