import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def plot_silhouette_scores(silhouette_df, output_path=None):
    """
    Plot silhouette scores vs distance threshold.
    
    Args:
        silhouette_df (pd.DataFrame): DataFrame with threshold and silhouette_score columns
        output_path (str): Path to save plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(silhouette_df['threshold'], silhouette_df['silhouette_score'], marker='o')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Distance Threshold')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()


def plot_umap_clusters(embeddings_df, n_embedding_cols, output_dir='plots'):
    """
    Generate UMAP visualizations for each product's clusters.
    
    Args:
        embeddings_df (pd.DataFrame): DataFrame with embeddings and cluster labels
        n_embedding_cols (int): Number of embedding columns
        output_dir (str): Directory to save plots
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Run: pip install umap-learn")
        return
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for product in embeddings_df['product'].unique():
        print(f"Generating UMAP for: {product}")
        
        product_mask = embeddings_df['product'] == product
        product_embeddings = embeddings_df.loc[product_mask].iloc[:, :n_embedding_cols].values
        cluster_labels = embeddings_df.loc[product_mask, 'agglomerative_cluster'].values
        
        # Get cluster names if available
        if 'cluster_name' in embeddings_df.columns:
            cluster_names = embeddings_df.loc[product_mask, 'cluster_name'].values
        else:
            cluster_names = cluster_labels
        
        # Run UMAP
        reducer = umap.UMAP(
            n_neighbors=15, 
            min_dist=0.1, 
            metric='cosine', 
            random_state=42,
            init='random'
        )
        embedding_2d = reducer.fit_transform(product_embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embedding_2d[:, 0], 
            embedding_2d[:, 1], 
            c=cluster_labels, 
            cmap='Spectral', 
            s=5, 
            alpha=0.6
        )
        
        # Create legend with cluster names (limit to first 50)
        unique_clusters = np.unique(cluster_labels)
        cluster_to_name = {}
        for cluster in unique_clusters:
            idx = np.where(cluster_labels == cluster)[0][0]
            name = cluster_names[idx]
            cluster_to_name[cluster] = name if not pd.isna(name) else "Unnamed"
        
        # Show only first 50 clusters in legend
        legend_elements = []
        for i, cluster in enumerate(sorted(unique_clusters)[:50]):
            color = plt.cm.Spectral(i / max(len(unique_clusters) - 1, 1))
            cluster_name_str = str(cluster_to_name[cluster])[:40]
            legend_elements.append(Patch(facecolor=color, label=f"{cluster}: {cluster_name_str}"))
        
        if len(unique_clusters) > 50:
            legend_elements.append(Patch(facecolor='white', label=f"... and {len(unique_clusters) - 50} more clusters"))
        
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        plt.title(f'UMAP projection of {product} Embeddings ({len(unique_clusters)} clusters)')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        
        # Save plot
        safe_product_name = product.replace("/", "_").replace(" ", "_")
        plot_path = f'{output_dir}/umap_{safe_product_name}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
        
        plt.show()


def plot_cluster_distribution(embeddings_df, output_path=None):
    """
    Plot distribution of clusters across products.
    
    Args:
        embeddings_df (pd.DataFrame): DataFrame with product and cluster info
        output_path (str): Path to save plot (optional)
    """
    plt.figure(figsize=(12, 6))
    
    cluster_counts = embeddings_df.groupby(['product', 'agglomerative_cluster']).size().reset_index(name='count')
    products = cluster_counts['product'].unique()
    
    for product in products:
        product_data = cluster_counts[cluster_counts['product'] == product]
        plt.bar(product_data['agglomerative_cluster'], product_data['count'], alpha=0.7, label=product)
    
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Complaints')
    plt.title('Cluster Distribution by Product')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()
