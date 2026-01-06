"""
Main execution script for complaint clustering pipeline.

This script orchestrates the complete pipeline:
1. Load and preprocess complaint data
2. Generate embeddings for narratives
3. Find optimal clustering thresholds
4. Cluster complaints by product
5. Generate cluster names using LLM
6. Visualize results

Usage:
    python main.py
    
Configuration:
    Edit the CONFIG dictionary below to customize the pipeline behavior.
"""

import os
import pandas as pd
from datetime import date, timedelta

from preprocess import preprocess_data
from embed_complaints import ComplaintEmbedder
from cluster_complaints import (
    find_best_thresholds_by_product, 
    cluster_by_product,
    compute_cluster_centroids,
    save_cluster_centroids,
    load_cluster_centroids,
    predict_clusters
)
from name_clusters import create_cluster_names_batch
from visualize_clusters import plot_silhouette_scores, plot_umap_clusters, plot_cluster_distribution


# ======================== CONFIGURATION ========================
CONFIG = {
    # Data preprocessing
    #'days_back': ,                    # Number of days back to load data
    'first_date': '2023-01-01',                  # Or specify date as 'YYYY-MM-DD' (overrides days_back)
    'companies': ['UNITED SERVICES AUTOMOBILE ASSOCIATION'],                   # List of companies or None for defaults
    
    # Embedding
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'batch_size': 32,
    'embeddings_path': 'data/complaint_embeddings.pkl',
    'append_mode': False,                 # True: add new complaints to existing, False: regenerate all
    
    # Clustering
    'thresholds': None,                  # List of thresholds or None for defaults
    'max_samples': 25000,
    'min_samples_per_subproduct': 30,   # Minimum samples required per sub-product (smaller ones grouped into 'Other')
    'recluster': True,                  # True: re-cluster all data, False: predict using existing centroids
    'centroids_path': 'model/cluster_centroids.pkl',  # Path to save/load cluster centroids
    
    # Cluster naming
    'naming_model': 'Qwen/Qwen3-Next-80B-A3B-Instruct',
    'n_samples_for_naming': 10,
    'force_rename': False,               # Force regeneration of cluster names
    
    # Pipeline control
    'skip_embeddings': False,            # Skip if embeddings file exists
    'skip_clustering': False,            # Skip if clustered file exists
    'skip_naming': False,                # Skip cluster naming
    'skip_visualization': False,         # Skip visualization generation
}
# ================================================================


def main():
    """Run the complete complaint clustering pipeline."""
    
    # Create output directories
    os.makedirs('output', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    print("="*80)
    print("COMPLAINT CLUSTERING PIPELINE")
    print("="*80)
     
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    if CONFIG['first_date']:
        first_date = pd.to_datetime(CONFIG['first_date']).date()
    else:
        first_date = date.today() - timedelta(days=CONFIG['days_back'])
    
    df = preprocess_data(first_date_to_keep=first_date, companies_to_keep=CONFIG['companies'])
    print(f"Loaded {len(df)} complaints")
    print(f"Products: {df['product'].nunique()}")
    print(f"Companies: {df['company'].nunique()}")
    
    # Create company suffix for filenames
    if CONFIG['companies']:
        company_suffix = '_' + '_'.join([c.lower().replace(' ', '_').replace(',', '')[:20] for c in CONFIG['companies']])
    else:
        company_suffix = ''
    print(f"Company suffix for files: {company_suffix}")
    
    # Generate embeddings
    embeddings_path = CONFIG['embeddings_path'].replace('.pkl', f"{company_suffix}.pkl")
    
    if CONFIG['skip_embeddings'] and os.path.exists(embeddings_path):
        print(f"\nLoading existing embeddings from {embeddings_path}...")
        embeddings_df = pd.read_pickle(embeddings_path)
        print(f"Loaded embeddings for {len(embeddings_df)} complaints")
        
        # Get number of embedding columns
        n_embedding_cols = len([col for col in embeddings_df.columns 
                               if isinstance(col, int)])
    elif CONFIG['append_mode'] and os.path.exists(embeddings_path):
        # Append mode: load existing embeddings and add new ones
        print(f"\nAppend mode: Loading existing embeddings from {embeddings_path}...")
        existing_embeddings_df = pd.read_pickle(embeddings_path)
        print(f"Loaded {len(existing_embeddings_df)} existing complaints")
        
        # Find new complaints not in existing data
        existing_ids = set(existing_embeddings_df['complaint_id'].values)
        new_complaints_mask = ~df['complaint_id'].isin(existing_ids)
        new_df = df[new_complaints_mask].reset_index(drop=True)
        
        if len(new_df) == 0:
            print("No new complaints found. Using existing embeddings.")
            embeddings_df = existing_embeddings_df
            n_embedding_cols = len([col for col in embeddings_df.columns 
                                   if isinstance(col, int)])
        else:
            print(f"Found {len(new_df)} new complaints to embed")
            print(f"\nGenerating embeddings for new complaints...")
            
            embedder = ComplaintEmbedder(model_name=CONFIG['model_name'])
            new_embeddings_array = embedder.embed_dataframe(new_df, batch_size=CONFIG['batch_size'])
            
            # Create DataFrame for new embeddings
            new_embeddings_df = pd.DataFrame(new_embeddings_array)
            new_embeddings_df['complaint_id'] = new_df['complaint_id'].values
            new_embeddings_df['product'] = new_df['product'].values
            new_embeddings_df['sub-product'] = new_df['sub-product'].values
            new_embeddings_df['company'] = new_df['company'].values
            new_embeddings_df['consumer_complaint_narrative'] = new_df['consumer_complaint_narrative'].values
            
            # Combine existing and new embeddings
            embeddings_df = pd.concat([existing_embeddings_df, new_embeddings_df], ignore_index=True)
            print(f"Total embeddings: {len(embeddings_df)} (existing: {len(existing_embeddings_df)}, new: {len(new_embeddings_df)})")
            
            # Save combined embeddings
            embeddings_df.to_pickle(embeddings_path)
            print(f"Saved combined embeddings to {embeddings_path}")
            
            n_embedding_cols = new_embeddings_array.shape[1]
    else:
        # Full regeneration mode
        print(f"\nGenerating embeddings for all complaints...")
        embedder = ComplaintEmbedder(model_name=CONFIG['model_name'])
        embeddings_array = embedder.embed_dataframe(df, batch_size=CONFIG['batch_size'])
        
        embeddings_df = embedder.save_embeddings(
            embeddings_array, 
            df, 
            embeddings_path,
            metadata_cols=['complaint_id', 'product', 'sub-product', 
                          'company', 'consumer_complaint_narrative']
        )
        n_embedding_cols = embeddings_array.shape[1]
    
    # Find optimal clustering thresholds per product
    thresholds_path = f'output/best_distance_thresholds{company_suffix}.csv'
    
    if CONFIG['skip_clustering'] and os.path.exists(thresholds_path):
        print(f"\nLoading existing thresholds from {thresholds_path}...")
        best_thresholds_df = pd.read_csv(thresholds_path)
    else:
        print(f"\nFinding optimal clustering thresholds...")
        best_thresholds_df = find_best_thresholds_by_product(
            embeddings_df,
            n_embedding_cols,
            thresholds=CONFIG['thresholds'],
            max_samples=CONFIG['max_samples']
        )
        
        best_thresholds_df.to_csv(thresholds_path, index=False)
        print(f"\nSaved best thresholds to {thresholds_path}")
        print(best_thresholds_df)
    
    # Cluster complaints by product
    clustered_path = f'data/complaint_embeddings_with_clusters{company_suffix}.pkl'
    centroids_path = CONFIG['centroids_path'].replace('.pkl', f"{company_suffix}.pkl")
    
    if CONFIG['skip_clustering'] and os.path.exists(clustered_path):
        print(f"\nLoading existing clusters from {clustered_path}...")
        embeddings_df = pd.read_pickle(clustered_path)
        
        # Verify cluster column exists
        if 'agglomerative_cluster' not in embeddings_df.columns:
            print(f"Warning: Loaded file missing cluster column. Will re-cluster.")
            CONFIG['skip_clustering'] = False
    
    if not CONFIG['skip_clustering']:
        if CONFIG['recluster'] or not os.path.exists(centroids_path):
            # Full re-clustering from scratch
            print(f"\nClustering complaints by product (full re-clustering)...")
            embeddings_df = cluster_by_product(
                embeddings_df,
                n_embedding_cols,
                best_thresholds_df,
                max_samples=CONFIG['max_samples']
            )
            
            # Compute and save cluster centroids for future use
            print(f"\nComputing cluster centroids...")
            centroids = compute_cluster_centroids(embeddings_df, n_embedding_cols)
            save_cluster_centroids(centroids, centroids_path)
        else:
            # Prediction mode: assign clusters to new or all complaints
            print(f"\nPredicting clusters using existing centroids from {centroids_path}...")
            centroids = load_cluster_centroids(centroids_path)
            
            # If in append mode, only predict for new complaints
            if CONFIG['append_mode'] and os.path.exists(clustered_path):
                # Load previously clustered data
                previous_clustered_df = pd.read_pickle(clustered_path)
                previous_ids = set(previous_clustered_df['complaint_id'].values)
                
                # Identify new complaints
                new_mask = ~embeddings_df['complaint_id'].isin(previous_ids)
                
                if new_mask.sum() > 0:
                    print(f"Predicting clusters for {new_mask.sum()} new complaints...")
                    new_complaints_df = embeddings_df[new_mask].copy()
                    new_complaints_df = predict_clusters(
                        new_complaints_df,
                        n_embedding_cols,
                        centroids
                    )
                    
                    # Combine with existing clustered data
                    embeddings_df = pd.concat([previous_clustered_df, new_complaints_df], ignore_index=True)
                    print(f"Combined: {len(embeddings_df)} total complaints ({len(previous_clustered_df)} existing + {len(new_complaints_df)} new)")
                else:
                    print("No new complaints to predict. Using existing clusters.")
                    embeddings_df = previous_clustered_df
            else:
                # Predict for all complaints
                embeddings_df = predict_clusters(
                    embeddings_df,
                    n_embedding_cols,
                    centroids
                )
        
        embeddings_df.to_pickle(clustered_path)
        print(f"Saved clustered data to {clustered_path}")
    
    # Generate cluster names
    cluster_names_path = f'output/cluster_names{company_suffix}.csv'
    
    if not CONFIG['skip_naming']:
        # Check if clustering has been done
        if 'agglomerative_cluster' not in embeddings_df.columns:
            print(f"\nWarning: No cluster labels found. Skipping cluster naming.")
            print("Please ensure clustering has been completed before naming clusters.")
        else:
            print(f"\nGenerating cluster names using LLM...")
            
            if os.path.exists(cluster_names_path) and not CONFIG['force_rename']:
                print(f"Loading existing cluster names from {cluster_names_path}")
                cluster_names_df = pd.read_csv(cluster_names_path)
                
                # Verify the CSV has the required columns
                required_cols = ['product', 'agglomerative_cluster']
                missing_cols = [col for col in required_cols if col not in cluster_names_df.columns]
                if missing_cols:
                    print(f"Warning: Cluster names file missing columns {missing_cols}. Regenerating.")
                    CONFIG['force_rename'] = True
                else:
                    # Check if we have names for all current product-cluster combinations
                    current_combos = set(
                        tuple(x) for x in embeddings_df[['product', 'agglomerative_cluster']].drop_duplicates().values
                    )
                    existing_combos = set(
                        tuple(x) for x in cluster_names_df[['product', 'agglomerative_cluster']].values
                    )
                    missing_combos = current_combos - existing_combos
                    
                    if missing_combos:
                        print(f"Warning: Cluster names file missing {len(missing_combos)} product-cluster combinations.")
                        print(f"Regenerating all cluster names to ensure consistency.")
                        CONFIG['force_rename'] = True
            
            if CONFIG['force_rename'] or not os.path.exists(cluster_names_path):
                cluster_names_list = []
                
                for product in embeddings_df['product'].unique():
                    print(f"\nNaming clusters for: {product}")
                    
                    # Sample narratives for this product
                    product_df = embeddings_df[embeddings_df['product'] == product]
                    
                    for cluster_id in product_df['agglomerative_cluster'].unique():
                        cluster_mask = product_df['agglomerative_cluster'] == cluster_id
                        sample_narratives = product_df[cluster_mask]['consumer_complaint_narrative'].dropna().sample(
                            n=min(CONFIG['n_samples_for_naming'], cluster_mask.sum()), 
                            random_state=42
                        ).tolist()
                        
                        from name_clusters import create_cluster_name
                        cluster_name = create_cluster_name(sample_narratives, model=CONFIG['naming_model'])
                        
                        cluster_names_list.append({
                            'product': product,
                            'agglomerative_cluster': cluster_id,
                            'cluster_name': cluster_name
                        })
                
                cluster_names_df = pd.DataFrame(cluster_names_list)
                cluster_names_df.to_csv(cluster_names_path, index=False)
                print(f"\nSaved cluster names to {cluster_names_path}")
            
            # Merge cluster names with embeddings (only if both have the column)
            if 'agglomerative_cluster' in cluster_names_df.columns:
                embeddings_df = embeddings_df.merge(
                    cluster_names_df,
                    on=['product', 'agglomerative_cluster'],
                    how='left'
                )
                
                # Save updated embeddings with cluster names
                embeddings_df.to_pickle(clustered_path)
            else:
                print("Warning: Could not merge cluster names - missing agglomerative_cluster column")
    else:
        print(f"\nSkipping cluster naming")
    
    # Visualize results
    if not CONFIG['skip_visualization']:
        print(f"\nGenerating visualizations...")
        
        # Prepare date range and companies for subtitle
        date_range = (df['date_received'].min(), df['date_received'].max()) if 'date_received' in df.columns else None
        companies_list = CONFIG['companies'] if CONFIG['companies'] else df['company'].unique().tolist()
        
        # Plot UMAP for each product-subproduct
        plot_umap_clusters(embeddings_df, n_embedding_cols, output_dir='plots', date_range=date_range, companies=companies_list)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nOutputs saved to:")
        print(f"  - Embeddings: {clustered_path}")
        print(f"  - Thresholds: {thresholds_path}")
        print(f"  - Centroids: {centroids_path}")
        if not CONFIG['skip_naming']:
            print(f"  - Cluster names: {cluster_names_path}")
        print(f"  - Visualizations: plots/")
    else:
        print(f"\nSkipping visualization")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total complaints: {len(embeddings_df)}")
    print(f"Total products: {embeddings_df['product'].nunique()}")
    print(f"Total clusters: {embeddings_df['agglomerative_cluster'].nunique()}")
    print("\nClusters per product:")
    clusters_per_product = embeddings_df.groupby('product')['agglomerative_cluster'].nunique().sort_values(ascending=False)
    print(clusters_per_product)
    
    return embeddings_df


if __name__ == "__main__":
    embeddings_df = main()
