import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def find_optimal_threshold(embeddings, thresholds=None, max_samples=25000, random_state=442):
    """
    Find optimal distance threshold for clustering using silhouette score.
    
    Args:
        embeddings (np.ndarray): Array of embeddings
        thresholds (list): List of thresholds to try
        max_samples (int): Maximum samples to use for evaluation
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with threshold, n_clusters, and silhouette scores
    """
    if thresholds is None:
        thresholds = [0.75, 1, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 10, 15, 20, 25, 28, 30, 35, 40, 45, 50]
    
    # Sample if needed
    if len(embeddings) > max_samples:
        embeddings_subset = resample(embeddings, n_samples=max_samples, random_state=random_state)
    else:
        embeddings_subset = embeddings
    
    results = []
    
    for thresh in thresholds:
        print(f"Testing threshold: {thresh}")
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=thresh)
        cluster_labels = clustering.fit_predict(embeddings_subset)
        n_clusters = len(set(cluster_labels))
        
        if n_clusters > 1 and n_clusters < len(embeddings_subset):
            sil_score = silhouette_score(embeddings_subset, cluster_labels)
            results.append({
                'threshold': thresh,
                'n_clusters': n_clusters,
                'silhouette_score': sil_score
            })
            print(f"  Clusters: {n_clusters}, Silhouette Score: {sil_score:.4f}")
        else:
            print(f"  Clusters: {n_clusters} (invalid for silhouette score)")
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        best_idx = results_df['silhouette_score'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_n_clusters = results_df.loc[best_idx, 'n_clusters']
        print(f"\nBest threshold: {best_threshold} (n_clusters={best_n_clusters})")
    
    return results_df


def compute_cluster_centroids(embeddings_df, n_embedding_cols, cluster_col='agglomerative_cluster'):
    """
    Compute centroids for each cluster within each product.
    
    Args:
        embeddings_df (pd.DataFrame): DataFrame with embeddings and cluster labels
        n_embedding_cols (int): Number of embedding columns
        cluster_col (str): Name of column containing cluster labels
        
    Returns:
        dict: Dictionary mapping (product, cluster_id) to centroid vector
    """
    centroids = {}
    
    for product in embeddings_df['product'].unique():
        product_df = embeddings_df[embeddings_df['product'] == product]
        
        for cluster_id in product_df[cluster_col].unique():
            cluster_mask = product_df[cluster_col] == cluster_id
            cluster_embeddings = product_df.loc[cluster_mask].iloc[:, :n_embedding_cols].values
            
            # Compute centroid as mean of all embeddings in cluster
            centroid = cluster_embeddings.mean(axis=0)
            centroids[(product, cluster_id)] = centroid
    
    return centroids


def save_cluster_centroids(centroids, output_path='model/cluster_centroids.pkl'):
    """
    Save cluster centroids to a pickle file.
    
    Args:
        centroids (dict): Dictionary of cluster centroids
        output_path (str): Path to save the centroids
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(centroids, f)
    print(f"Saved cluster centroids to {output_path}")


def load_cluster_centroids(input_path='model/cluster_centroids.pkl'):
    """
    Load cluster centroids from a pickle file.
    
    Args:
        input_path (str): Path to load the centroids from
        
    Returns:
        dict: Dictionary of cluster centroids
    """
    with open(input_path, 'rb') as f:
        centroids = pickle.load(f)
    print(f"Loaded cluster centroids from {input_path}")
    return centroids


def predict_clusters(embeddings_df, n_embedding_cols, centroids, cluster_col='agglomerative_cluster'):
    """
    Assign new complaints to existing clusters based on cosine similarity to centroids.
    
    Args:
        embeddings_df (pd.DataFrame): DataFrame with embeddings and product info
        n_embedding_cols (int): Number of embedding columns
        centroids (dict): Dictionary mapping (product, cluster_id) to centroid vectors
        cluster_col (str): Name of column to store cluster labels
        
    Returns:
        pd.DataFrame: DataFrame with predicted cluster labels
    """
    embeddings_df = embeddings_df.copy()
    
    for product in embeddings_df['product'].unique():
        product_mask = embeddings_df['product'] == product
        product_embeddings = embeddings_df.loc[product_mask].iloc[:, :n_embedding_cols].values
        
        # Get all centroids for this product
        product_centroids = {k: v for k, v in centroids.items() if k[0] == product}
        
        if len(product_centroids) == 0:
            print(f"Warning: No centroids found for {product}, skipping")
            continue
        
        # Create array of centroids and corresponding cluster IDs
        cluster_ids = [k[1] for k in product_centroids.keys()]
        centroid_vectors = np.array([v for v in product_centroids.values()])
        
        # Compute cosine similarity between each embedding and all centroids
        # Shape: (n_embeddings, n_clusters)
        similarities = cosine_similarity(product_embeddings, centroid_vectors)
        
        # Assign to cluster with highest similarity
        closest_cluster_indices = similarities.argmax(axis=1)
        assigned_clusters = [cluster_ids[idx] for idx in closest_cluster_indices]
        
        embeddings_df.loc[product_mask, cluster_col] = assigned_clusters
        print(f"Assigned {product_mask.sum()} complaints to {len(cluster_ids)} clusters for {product}")
    
    return embeddings_df


def cluster_by_product(embeddings_df, n_embedding_cols, best_thresholds_df, 
                       cluster_col='agglomerative_cluster', max_samples=25000):
    """
    Cluster embeddings separately for each product.
    
    Args:
        embeddings_df (pd.DataFrame): DataFrame with embeddings and product info
        n_embedding_cols (int): Number of embedding columns
        best_thresholds_df (pd.DataFrame): DataFrame with best thresholds per product
        cluster_col (str): Name of column to store cluster labels
        max_samples (int): Maximum samples to cluster before using KNN for remaining
        
    Returns:
        pd.DataFrame: DataFrame with cluster labels added
    """
    embeddings_df = embeddings_df.copy()
    
    for product in embeddings_df['product'].unique():
        product_mask = embeddings_df['product'] == product
        n_product_complaints = product_mask.sum()
        
        # Get number of clusters for this product
        product_thresholds = best_thresholds_df[best_thresholds_df['product'] == product]
        if len(product_thresholds) == 0:
            print(f"Warning: No threshold found for {product}, skipping")
            continue
            
        n_clusters = int(product_thresholds['num_clusters'].values[0])
        print(f"Clustering {product}: {n_product_complaints} complaints into {n_clusters} clusters")
        
        if n_product_complaints > max_samples:
            # Sample for clustering
            sampled_indices = embeddings_df.loc[product_mask].sample(max_samples, random_state=442).index
            product_embeddings_sample = embeddings_df.loc[sampled_indices].iloc[:, :n_embedding_cols].values
            
            # Cluster the sample
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels_sample = clustering.fit_predict(product_embeddings_sample)
            
            # Assign cluster labels to sampled points
            embeddings_df.loc[sampled_indices, cluster_col] = cluster_labels_sample
            
            # For remaining points, assign to nearest cluster using KNN
            remaining_mask = product_mask & ~embeddings_df.index.isin(sampled_indices)
            if remaining_mask.sum() > 0:
                remaining_embeddings = embeddings_df.loc[remaining_mask].iloc[:, :n_embedding_cols].values
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(product_embeddings_sample)
                distances, indices = nn.kneighbors(remaining_embeddings)
                remaining_cluster_labels = cluster_labels_sample[indices.flatten()]
                embeddings_df.loc[remaining_mask, cluster_col] = remaining_cluster_labels
        else:
            # Cluster all points if under threshold
            product_embeddings = embeddings_df.loc[product_mask].iloc[:, :n_embedding_cols].values
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering.fit_predict(product_embeddings)
            embeddings_df.loc[product_mask, cluster_col] = cluster_labels
    
    return embeddings_df


def find_best_thresholds_by_product(embeddings_df, n_embedding_cols, 
                                   thresholds=None, max_samples=25000):
    """
    Find optimal thresholds for each product separately.
    
    Args:
        embeddings_df (pd.DataFrame): DataFrame with embeddings and product info
        n_embedding_cols (int): Number of embedding columns
        thresholds (list): List of thresholds to try
        max_samples (int): Maximum samples per product for evaluation
        
    Returns:
        pd.DataFrame: DataFrame with best threshold and n_clusters per product
    """
    results = []
    
    for product in embeddings_df['product'].unique():
        print(f"\n{'='*60}")
        print(f"Finding optimal threshold for: {product}")
        print(f"{'='*60}")
        
        product_embeddings = embeddings_df[embeddings_df['product'] == product].iloc[:, :n_embedding_cols].values
        
        product_results = find_optimal_threshold(
            product_embeddings, 
            thresholds=thresholds,
            max_samples=max_samples
        )
        
        if len(product_results) > 0:
            best_idx = product_results['silhouette_score'].idxmax()
            results.append({
                'product': product,
                'best_distance_threshold': product_results.loc[best_idx, 'threshold'],
                'num_clusters': product_results.loc[best_idx, 'n_clusters'],
                'silhouette_score': product_results.loc[best_idx, 'silhouette_score']
            })
    
    return pd.DataFrame(results)
