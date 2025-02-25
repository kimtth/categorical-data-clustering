import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


CATEGORY_COLUMN = "category"

def filtered_df_overlap(df):
    """Remove columns that are not needed for the analysis."""
    excluded_cols = [
        "ProductID",
    ]
    return df.drop(excluded_cols, axis=1, errors="ignore")


def filter_categorical_columns(df):
    """Keep only categorical columns."""
    return df.select_dtypes(include=["object"])


def one_hot_encode(df):
    """One-hot encode the data for distance computation."""
    df = df.astype(str)
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(df)
    return X_encoded


def load_and_prepare_data(csv_file, sample_size=100):
    df = pd.read_csv(csv_file)

    # Remove rows/columns with too many missing values
    na_threshold_rows = 0.4
    na_threshold_cols = 0.7
    df = df.dropna(thresh=na_threshold_rows * len(df.columns))
    df = df.dropna(thresh=na_threshold_cols * len(df), axis=1)
    df = df.fillna("")
    df = df.reset_index(drop=True)

    # Keep only categorical columns
    df = filter_categorical_columns(df)

    # Take a sample of the data
    df_sample = df.head(sample_size)

    # Extract and encode cluster labels from CATEGORY_COLUMN
    if CATEGORY_COLUMN in df.columns:
        # 1. When the cluster labels are numeric, use the following code
        # label_encoder = LabelEncoder()
        # cluster_labels = label_encoder.fit_transform(df[CATEGORY_COLUMN])
        # 2. The cluster_labels should be the original string labels
        cluster_labels = df[CATEGORY_COLUMN].values
    else:
        print("Column 'category' not found. Cluster labels will be None.")
        cluster_labels = None

    # Clean the sample data and save for inspection
    df_sample_clean = filtered_df_overlap(df_sample)
    df_sample_clean.to_csv("df_sample.csv", encoding="utf8", index=False)

    return df, cluster_labels


def compute_distance_matrix(df, distance_mode="jaccard", recalc=True):
    if recalc:
        df_encoded = one_hot_encode(filtered_df_overlap(df))
        if distance_mode == "jaccard":
            dist_matrix = pairwise_distances(df_encoded, metric="jaccard")
        elif distance_mode == "hamming":
            dist_matrix = pairwise_distances(df_encoded, metric="hamming")
        elif distance_mode == "overlap":
            dist_matrix = squareform(
                pdist(df_encoded, metric=lambda a, b: np.sum(a == b) / len(a))
            )
        elif distance_mode == "gower":
            dist_matrix = gower_distance(df_encoded)
        else:
            raise ValueError(f"Unsupported distance mode: {distance_mode}")
        pickle.dump(dist_matrix, open(f"{distance_mode}_dist_matrix.pkl", "wb"))
    else:
        dist_matrix = pickle.load(open(f"{distance_mode}_dist_matrix.pkl", "rb"))
    return dist_matrix


def gower_distance(df_encoded):
    df_str = pd.DataFrame(df_encoded).astype(str)
    dist_matrix = squareform(pdist(df_str, metric=lambda u, v: np.mean(u != v)))
    return dist_matrix


def perform_tsne_embedding(dist_matrix):
    tsne_2d = TSNE(
        n_components=2,
        max_iter=1000,
        perplexity=3,
        learning_rate="auto",
        metric="precomputed",
        random_state=42,
        init="random",
    )
    tsne_results_2d = tsne_2d.fit_transform(dist_matrix)
    tsne_3d = TSNE(
        n_components=3,
        max_iter=1000,
        perplexity=3,
        learning_rate="auto",
        metric="precomputed",
        random_state=42,
        init="random",
    )
    tsne_results_3d = tsne_3d.fit_transform(dist_matrix)
    return tsne_results_2d, tsne_results_3d


def perform_pca(dist_matrix):
    """Perform PCA on the distance matrix."""
    pca = PCA(n_components=2)
    return pca.fit_transform(dist_matrix)


def plot_embeddings(tsne_2d, tsne_3d, pca_2d, cluster_labels=None):
    fig = plt.figure(figsize=(15, 5))

    # If string cluster_labels are provided, create a mapping to colors.
    if cluster_labels is not None:
        unique_labels = np.unique(cluster_labels)
        cmap = plt.get_cmap("viridis", len(unique_labels))
        color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}
        # Create a color for each point based on its string label.
        colors = [color_dict[label] for label in cluster_labels]
    else:
        colors = None

    # 2D T-SNE plot
    ax1 = fig.add_subplot(131)
    ax1.scatter(tsne_2d[:, 0], tsne_2d[:, 1], color=colors)
    ax1.set_title("2D T-SNE")
    if cluster_labels is not None:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_dict[label],
                label=label,
                markersize=8,
            )
            for label in unique_labels
        ]
        ax1.legend(handles=handles, title=CATEGORY_COLUMN)

    # 3D T-SNE plot
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.scatter(tsne_3d[:, 0], tsne_3d[:, 1], tsne_3d[:, 2], color=colors)
    ax2.set_title("3D T-SNE")
    if cluster_labels is not None:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_dict[label],
                label=label,
                markersize=8,
            )
            for label in unique_labels
        ]
        ax2.legend(handles=handles, title=CATEGORY_COLUMN)

    # 2D PCA plot
    ax3 = fig.add_subplot(133)
    ax3.scatter(pca_2d[:, 0], pca_2d[:, 1], color=colors)
    ax3.set_title("2D PCA")
    if cluster_labels is not None:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_dict[label],
                label=label,
                markersize=8,
            )
            for label in unique_labels
        ]
        ax3.legend(handles=handles, title=CATEGORY_COLUMN)

    plt.tight_layout()
    plt.show(block=True)


def main(csv_file_path, re_calc=True, distance_mode="jaccard"):
    # Load data and get cluster labels based on CATEGORY_COLUMN
    df, cluster_labels = load_and_prepare_data(csv_file_path)

    # Compute the distance matrix using the specified metric
    dist_matrix = compute_distance_matrix(df, distance_mode, recalc=re_calc)

    # Perform embeddings using T-SNE and PCA
    tsne_2d, tsne_3d = perform_tsne_embedding(dist_matrix)
    pca_2d = perform_pca(dist_matrix)

    # Plot the results, coloring the points by the cluster labels (if available)
    plot_embeddings(tsne_2d, tsne_3d, pca_2d, cluster_labels)


if __name__ == "__main__":
    main(
        csv_file_path="dataset.csv",  # Adjust the path as needed
        re_calc=True,
        distance_mode="jaccard",  # Change distance mode if required
    )
