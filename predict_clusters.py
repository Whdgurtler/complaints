"""
Predict clusters for new complaints using existing cluster centroids.

This script allows you to assign new complaints to existing clusters without
re-running the full clustering pipeline.

Usage:
    python predict_clusters.py
    
Configuration:
    Edit the CONFIG dictionary below to customize behavior.
"""

import os
import pandas as pd
from datetime import date, timedelta

from preprocess import preprocess_data
from embed_complaints import ComplaintEmbedder
from cluster_complaints import load_cluster_centroids, predict_clusters


# ======================== CONFIGURATION ========================
CONFIG = {
    # Data preprocessing
    'days_back': 7,                      # Number of days back to load NEW data
    'first_date': None,                  # Or specify date as 'YYYY-MM-DD'
    'companies': None,                   # List of companies or None for defaults
    
    # Embedding
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'batch_size': 32,
    
    # Paths
    'centroids_path': 'model/cluster_centroids.pkl',
    'cluster_names_path': 'output/cluster_names.csv',
    'output_path': 'data/new_complaints_clustered.pkl',
}
# ================================================================


def main():
    """Predict clusters for new complaints."""
    
    print("="*80)
    print("PREDICT CLUSTERS FOR NEW COMPLAINTS")
    print("="*80)
    
    # Load and preprocess new data
    print("\nLoading new complaint data...")
    if CONFIG['first_date']:
        first_date = pd.to_datetime(CONFIG['first_date']).date()
    else:
        first_date = date.today() - timedelta(days=CONFIG['days_back'])
    
    df = preprocess_data(first_date_to_keep=first_date, companies_to_keep=CONFIG['companies'])
    print(f"Loaded {len(df)} new complaints")
    print(f"Products: {df['product'].nunique()}")
    print(f"Companies: {df['company'].nunique()}")
    
    # Generate embeddings for new data
    print(f"\nGenerating embeddings for new complaints...")
    embedder = ComplaintEmbedder(model_name=CONFIG['model_name'])
    embeddings_array = embedder.embed_dataframe(df, batch_size=CONFIG['batch_size'])
    
    # Create embeddings DataFrame
    embeddings_df = pd.DataFrame(embeddings_array)
    embeddings_df['complaint_id'] = df['complaint_id'].values
    embeddings_df['date_received'] = df['date_received'].values
    embeddings_df['company'] = df['company'].values
    embeddings_df['product'] = df['product'].values
    embeddings_df['sub-product'] = df['sub-product'].values
    embeddings_df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].values
    
    n_embedding_cols = embeddings_array.shape[1]
    
    # Load existing cluster centroids
    print(f"\nLoading cluster centroids from {CONFIG['centroids_path']}...")
    if not os.path.exists(CONFIG['centroids_path']):
        print(f"ERROR: Centroids file not found at {CONFIG['centroids_path']}")
        print("Please run the full pipeline first with 'recluster=True' to create centroids.")
        return None
    
    centroids = load_cluster_centroids(CONFIG['centroids_path'])
    
    # Predict clusters
    print(f"\nPredicting clusters for new complaints...")
    embeddings_df = predict_clusters(
        embeddings_df,
        n_embedding_cols,
        centroids
    )
    
    # Load cluster names if available
    if os.path.exists(CONFIG['cluster_names_path']):
        print(f"\nMerging cluster names from {CONFIG['cluster_names_path']}...")
        cluster_names_df = pd.read_csv(CONFIG['cluster_names_path'])
        embeddings_df = embeddings_df.merge(
            cluster_names_df,
            on=['product', 'agglomerative_cluster'],
            how='left'
        )
    
    # Save results
    embeddings_df.to_pickle(CONFIG['output_path'])
    print(f"\nSaved predictions to {CONFIG['output_path']}")
    
    # Print summary
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"Total complaints predicted: {len(embeddings_df)}")
    print(f"Total products: {embeddings_df['product'].nunique()}")
    print(f"Total clusters assigned: {embeddings_df['agglomerative_cluster'].nunique()}")
    
    print("\nCluster distribution by product:")
    cluster_dist = embeddings_df.groupby(['product', 'agglomerative_cluster']).size().reset_index(name='count')
    for product in cluster_dist['product'].unique():
        product_clusters = cluster_dist[cluster_dist['product'] == product]
        print(f"\n{product}:")
        for _, row in product_clusters.head(10).iterrows():
            cluster_id = row['agglomerative_cluster']
            count = row['count']
            if 'cluster_name' in embeddings_df.columns:
                name = embeddings_df[
                    (embeddings_df['product'] == product) & 
                    (embeddings_df['agglomerative_cluster'] == cluster_id)
                ]['cluster_name'].iloc[0]
                print(f"  Cluster {cluster_id} ({name}): {count} complaints")
            else:
                print(f"  Cluster {cluster_id}: {count} complaints")
    
    return embeddings_df


if __name__ == "__main__":
    result_df = main()
