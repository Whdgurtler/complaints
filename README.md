# Complaint Clustering Pipeline

A modular pipeline for clustering and analyzing consumer complaints using embeddings and machine learning.

## Overview

This pipeline processes consumer complaint narratives through the following stages:
1. **Data Preprocessing** - Filter and clean complaint data
2. **Embedding Generation** - Convert narratives to vector embeddings
3. **Clustering** - Group similar complaints using agglomerative clustering
4. **Cluster Naming** - Generate descriptive names using LLM
5. **Visualization** - Create UMAP visualizations of clusters

### Two Clustering Modes

**Full Clustering (recluster=True)**
- Runs agglomerative clustering from scratch on all data
- Computes and saves cluster centroids for future predictions
- Use for initial setup or when cluster definitions need updating

**Prediction Mode (recluster=False)**
- Assigns new complaints to existing clusters using cosine similarity
- Fast and efficient for incremental updates
- Maintains consistent cluster definitions over time

### Two Embedding Modes

**Full Regeneration (append_mode=False)**
- Generates embeddings for all data from scratch
- Use for initial setup or when reprocessing everything

**Append Mode (append_mode=True)**
- Adds only new complaints to existing embeddings
- Automatically detects which complaints are new (by complaint_id)
- Combines with existing data for unified analysis
- Much faster for daily/weekly updates

## Project Structure

```
├── main.py                    # Main execution script
├── predict_clusters.py        # Standalone prediction script for new data
├── preprocess.py              # Data preprocessing functions
├── embed_complaints.py        # Embedding generation module
├── cluster_complaints.py      # Clustering and prediction algorithms
├── name_clusters.py           # LLM-based cluster naming
├── visualize_clusters.py      # Visualization functions
├── data_load.py              # Data loading utilities
├── data/                     # Data directory (embeddings, raw data)
├── model/                    # Model artifacts (cluster centroids)
├── output/                   # Output files (thresholds, cluster names)
└── plots/                    # Generated visualizations
```

## Installation

### Requirements

```bash
pip install pandas numpy scikit-learn torch transformers huggingface_hub matplotlib umap-learn
```

### GPU Support (Optional but Recommended)

For faster embedding generation, install CUDA-enabled PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Initial Setup: Full Clustering

First time setup - creates clusters and saves centroids:

```python
# Edit CONFIG in main.py:
CONFIG = {
    'days_back': 365,
    'append_mode': False,   # Generate all embeddings
    'recluster': True,      # Full clustering
    'skip_embeddings': False,
    # ... other settings
}
```

```bash
python main.py
```

This will:
- Load and preprocess data
- Generate embeddings for all complaints
- Find optimal clustering thresholds
- Cluster all complaints
- **Save cluster centroids** to `model/cluster_centroids.pkl`
- Generate cluster names
- Create visualizations

### Incremental Updates: Append + Predict Mode

Add new complaints to existing model (fast):

```python
# Edit CONFIG in main.py:
CONFIG = {
    'days_back': 7,         # Only load recent data
    'append_mode': True,    # Add new embeddings to existing
    'recluster': False,     # Predict using existing centroids
    'skip_embeddings': False,
    # ... other settings
}
```

```bash
python main.py
```

This will:
- Load only recent complaint data
- **Detect which complaints are new** (not in existing embeddings)
- Generate embeddings only for new complaints
- **Append to existing embeddings**
- **Assign new complaints to existing clusters** using cosine similarity
- Merge cluster names
- **Update visualizations with all data** (old + new)
- Save results

### Common Workflow

**Week 1: Initial Setup**
```python
CONFIG = {'append_mode': False, 'recluster': True, 'days_back': 365}
```
```bash
python main.py
```

**Week 2-8: Incremental Updates**
```python
CONFIG = {'append_mode': True, 'recluster': False, 'days_back': 7}
```
```bash
python main.py  # Runs daily/weekly
```

**Month 2: Periodic Re-clustering**
```python
CONFIG = {'append_mode': True, 'recluster': True, 'days_back': 30}
```
```bash
python main.py  # Re-establish clusters with all accumulated data
```

## Module Documentation

### preprocess.py

**`preprocess_data(first_date_to_keep, companies_to_keep)`**
- Loads and preprocesses complaint data
- Filters by date and company
- Removes missing narratives and outliers

### embed_complaints.py

**`ComplaintEmbedder`** - Main class for generating embeddings
- `embed_text(texts)` - Generate embeddings for text batch
- `embed_dataframe(df, text_column, batch_size)` - Embed entire DataFrame
- `save_embeddings(embeddings_array, df, output_path)` - Save with metadata

### cluster_complaints.py

**`find_optimal_threshold(embeddings, thresholds)`**
- Tests multiple distance thresholds
- Returns silhouette scores for each

**`find_best_thresholds_by_product(embeddings_df, n_embedding_cols)`**
- Finds optimal threshold per product
- Returns DataFrame with best settings

**`cluster_by_product(embeddings_df, n_embedding_cols, best_thresholds_df)`**
- Clusters each product separately (full clustering)
- Uses KNN for large datasets

**`compute_cluster_centroids(embeddings_df, n_embedding_cols)`**
- Computes centroid (mean) for each cluster
- Returns dictionary mapping (product, cluster_id) to centroid vector

**`save_cluster_centroids(centroids, output_path)`**
- Saves centroids to pickle file

**`load_cluster_centroids(input_path)`**
- Loads centroids from pickle file

**`predict_clusters(embeddings_df, n_embedding_cols, centroids)`**
- Assigns complaints to existing clusters
- Uses cosine similarity to find nearest cluster
- Fast prediction without re-clustering

### name_clusters.py

**`sample_narratives_by_product_cluster(df, product, cluster_col, n_samples)`**
- Samples representative narratives from each cluster

**`create_cluster_name(sample_narratives, model, client)`**
- Generates descriptive cluster name using LLM

**`create_cluster_names_batch(df, product, cluster_col, n_samples, model)`**
- Efficiently names all clusters for a product

### visualize_clusters.py

**`plot_silhouette_scores(silhouette_df, output_path)`**
- Plots silhouette score vs threshold

**`plot_umap_clusters(embeddings_df, n_embedding_cols, output_dir)`**
- Generates UMAP visualizations per product

**`plot_cluster_distribution(embeddings_df, output_path)`**
- Shows cluster size distribution

## Output Files

After running the pipeline, you'll find:

### model/
- `cluster_centroids.pkl` - Saved cluster centroids for prediction

### data/
- `complaint_embeddings.pkl` - Raw embeddings with metadata
- `complaint_embeddings_with_clusters.pkl` - Embeddings with cluster labels and names
- `new_complaints_clustered.pkl` - Predictions from predict_clusters.py

### output/
- `best_distance_thresholds.csv` - Optimal thresholds per product
- `cluster_names.csv` - Generated cluster names

### plots/
- `umap_<product_name>.png` - UMAP visualization for each product

## Example Workflows

### Workflow 1: Initial Setup
```bash
# Edit main.py CONFIG: recluster=True, days_back=365
python main.py
```

### Workflow 2: Daily/Weekly Predictions
```bash
# Edit predict_clusters.py CONFIG: days_back=7
python predict_clusters.py
```

### Workflow 3: Monthly Re-clustering
```bash
# Edit main.py CONFIG: recluster=True, days_back=30
python main.py
```

### Workflow 4: Update Only Cluster Names
```python
# Edit main.py CONFIG:
CONFIG = {
    'skip_embeddings': True,
    'skip_clustering': True,
    'force_rename': True,
    # ...
}
```

```bash
python main.py
```

## Tips

1. **First time running**: Use `append_mode=False, recluster=True` to establish initial model
2. **Daily/weekly updates**: Use `append_mode=True, recluster=False` for fast incremental updates
3. **Ongoing model**: All visualizations automatically include old + new data in append mode
4. **No duplicate work**: Append mode only processes new complaints not in existing embeddings
5. **GPU acceleration**: Ensure CUDA is available for faster embedding generation
6. **Memory management**: Use `max_samples` to limit memory usage for large datasets
7. **API costs**: Use `skip_naming` during development to avoid LLM API calls
8. **Re-clustering**: Periodically re-cluster (monthly/quarterly) to adapt to changing patterns

## Configuration Quick Reference

| Scenario | append_mode | recluster | Use Case |
|----------|------------|-----------|----------|
| Initial setup | False | True | First time, create everything |
| Daily updates | True | False | Add new data, fast predictions |
| Weekly updates | True | False | Add new data, fast predictions |
| Monthly refresh | True | True | Re-cluster all accumulated data |
| Full rebuild | False | True | Start fresh, regenerate everything |

## Performance Notes

### Full Clustering (recluster=True)
- **Embedding generation**: ~1-5 minutes per 10k complaints (GPU)
- **Clustering**: ~2-10 minutes per product depending on size
- **Cluster naming**: ~1-3 seconds per cluster (depends on API)
- **Visualization**: ~30-60 seconds per product

Total runtime for 1 year of data (~50k complaints): **15-30 minutes**

### Prediction Mode (recluster=False)
- **Embedding generation**: ~1-5 minutes per 10k new complaints (GPU)
- **Prediction**: ~1-5 seconds for 10k complaints
- **No clustering or naming needed**

Total runtime for 1 week of new data (~5k complaints) in **append mode**: **1-3 minutes**

**Speedup with append mode: 10-20x faster** for incremental updates!

## Real-World Example

**Day 1 (Initial Setup):**
```python
CONFIG = {'append_mode': False, 'recluster': True, 'days_back': 365}
# Result: 50,000 complaints processed in ~20 minutes
```

**Day 8 (Weekly Update):**
```python
CONFIG = {'append_mode': True, 'recluster': False, 'days_back': 7}
# Result: 2,000 new complaints added in ~2 minutes
# Total dataset now: 52,000 complaints
# Visualizations show all 52,000 complaints
```

**Day 15 (Weekly Update):**
```python
CONFIG = {'append_mode': True, 'recluster': False, 'days_back': 7}
# Result: 1,800 new complaints added in ~2 minutes
# Total dataset now: 53,800 complaints
```

**Day 30 (Monthly Re-cluster):**
```python
CONFIG = {'append_mode': True, 'recluster': True, 'days_back': 30}
# Result: Re-clusters all 53,800 complaints in ~22 minutes
# Updates centroids to reflect new patterns
```
