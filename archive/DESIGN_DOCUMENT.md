# Complaints Data Processing and Clustering System - Design Document

## 1. System Overview

This system processes consumer complaint narratives from financial institutions, generates embeddings using transformer models, and clusters similar complaints to identify common themes and issues.

### 1.1 Purpose
- Analyze consumer complaints from major financial institutions
- Identify patterns and common complaint themes through machine learning clustering
- Enable data-driven insights for complaint categorization and trend analysis

### 1.2 Key Technologies
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: PyTorch, Transformers (HuggingFace), scikit-learn
- **Embeddings**: sentence-transformers/all-miniLM-L6-v2
- **Clustering**: Agglomerative Clustering with distance thresholds
- **Visualization**: UMAP, Matplotlib

---

## 2. Data Pipeline Architecture

### 2.1 Data Loading and Preprocessing

#### Input Data
- **Source**: `../data/complaints.csv`
- **Time Range**: Data from November 1, 2024 onwards (last 3 years filter applied)
- **Target Companies** (15 major financial institutions):
  - Capital One Financial Corporation
  - JPMorgan Chase & Co.
  - Block, Inc.
  - Wells Fargo & Company
  - Bank of America, National Association
  - Citibank, N.A.
  - Early Warning Services, LLC
  - Navy Federal Credit Union
  - Synchrony Financial
  - American Express Company
  - Discover Bank
  - PayPal Holdings, Inc
  - U.S. Bancorp
  - Chime Financial Inc
  - Ally Financial Inc.

#### Preprocessing Steps
1. **Column Normalization**: Lowercase column names and replace spaces with underscores
2. **Date Conversion**: Convert `date_received` to datetime and extract date only
3. **Temporal Filtering**: Keep only complaints from the last 3 years (≥ 2024-11-01)
4. **Company Filtering**: Filter to 15 target financial institutions
5. **Narrative Filtering**: 
   - Keep only complaints with non-null narratives
   - Remove narratives exceeding 4,930 characters (99th percentile)
6. **Product Filtering**: Remove rare product/sub-product combinations (<2% of total complaints)

### 2.2 Text Embedding Generation

#### Model Configuration
- **Model**: `sentence-transformers/all-miniLM-L6-v2`
- **Device**: GPU (CUDA) if available, otherwise CPU
- **Embedding Dimension**: 384 (default for this model)

#### Embedding Process
```
Input: Consumer complaint narrative text
↓
Tokenization (with padding and truncation)
↓
Model Forward Pass (with torch.no_grad())
↓
Mean Pooling over hidden states
↓
Output: 384-dimensional embedding vector
```

#### Batch Processing
- **Batch Size**: 32 narratives per batch
- **Output Format**: Pickle file containing:
  - `complaint_id`: Unique identifier
  - `product`: Product category
  - `sub-product`: Sub-product category
  - `company`: Financial institution name
  - `narrative`: Original complaint text
  - `embeddings`: NumPy array of embedding vectors

---

## 3. Clustering Architecture

### 3.1 Clustering Strategy

The system employs a **hierarchical agglomerative clustering** approach with product-specific optimization.

#### Phase 1: Optimal Threshold Discovery
For each product category:
1. **Sampling**: Use 25,000 samples if dataset exceeds this size
2. **Threshold Testing**: Test multiple distance thresholds [0.75, 1, 2.0, ..., 50]
3. **Evaluation**: Calculate silhouette score for each threshold
4. **Selection**: Choose threshold with highest silhouette score
5. **Output**: `best_distance_thresholds.csv` per product

#### Phase 2: Full Dataset Clustering
For each product:
- **Small Datasets** (≤ 25,000 complaints):
  - Cluster all points directly using optimal n_clusters
  
- **Large Datasets** (> 25,000 complaints):
  - Sample 25,000 points for clustering
  - Use agglomerative clustering with optimal n_clusters
  - Assign remaining points to nearest cluster using KNN (k=1)

### 3.2 Clustering Parameters

```python
AgglomerativeClustering(
    n_clusters=optimal_n_clusters,  # Determined per product
    random_state=442
)
```

### 3.3 Output Data Structure

**File**: `complaint_embeddings_with_clusters.pkl`
- All 384 embedding dimensions (columns 0-383)
- `complaint_id`: Unique identifier
- `product`: Product category
- `sub-product`: Sub-product category
- `company`: Financial institution
- `narrative`: Original complaint text
- `agglomerative_cluster`: Assigned cluster label

---

## 4. Cluster Naming and Interpretation

### 4.1 Automated Cluster Naming
The system uses the `name_clusters.py` module to generate human-readable cluster names:

1. **Sample Selection**: Extract 10 representative narratives per cluster
2. **Name Generation**: Use `create_cluster_name()` function to generate descriptive label
3. **Output**: `cluster_names.csv` with product, cluster, and name mappings

### 4.2 Cluster Metadata
- **File**: `output/cluster_names.csv`
- **Fields**:
  - `product`: Product category
  - `agglomerative_cluster`: Cluster ID
  - `cluster_name`: Human-readable description
  - `distance_threshold`: Optimal threshold used

---

## 5. Visualization

### 5.1 UMAP Dimensionality Reduction

**Purpose**: Visualize high-dimensional embeddings in 2D space

**Configuration**:
```python
umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
    init='random'
)
```

### 5.2 Visualization Output
- **Separate plot per product category**
- **File naming**: `plots/umap_{product_name}.png`
- **Features**:
  - Color-coded by cluster
  - Legend with cluster names (first 50 clusters shown)
  - 150 DPI resolution
  - Spectral colormap for cluster differentiation

---

## 6. Data Artifacts

### 6.1 Input Files
| File | Purpose |
|------|---------|
| `../data/complaints.csv` | Raw complaint data |

### 6.2 Intermediate Files
| File | Purpose |
|------|---------|
| `../data/complaint_embeddings.pkl` | Embeddings with metadata |
| `../data/silhouette_scores.csv` | Threshold evaluation results |
| `output/best_distance_thresholds.csv` | Optimal thresholds per product |

### 6.3 Output Files
| File | Purpose |
|------|---------|
| `../data/complaint_embeddings_with_clusters.pkl` | Final clustered embeddings |
| `output/cluster_names.csv` | Human-readable cluster labels |
| `plots/umap_{product}.png` | Visualization per product |

---

## 7. System Requirements

### 7.1 Hardware
- **GPU**: CUDA-compatible GPU recommended for embedding generation
- **Memory**: Sufficient RAM for handling 25,000+ embeddings (384 dimensions each)
- **Storage**: Space for embeddings pickle files (~100MB-1GB depending on dataset size)

### 7.2 Software Dependencies
```
pandas
numpy
torch
transformers
scikit-learn
umap-learn
matplotlib
```

### 7.3 Python Environment
- Python 3.7+
- CUDA toolkit (if using GPU acceleration)

---

## 8. Processing Workflow

```
┌─────────────────────────┐
│  Load Raw Complaint     │
│  Data (complaints.csv)  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Preprocess & Filter    │
│  - Normalize columns    │
│  - Filter dates         │
│  - Filter companies     │
│  - Filter narratives    │
│  - Remove rare products │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Generate Embeddings    │
│  - Load transformer     │
│  - Batch process (32)   │
│  - Save to pickle       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Optimize Clustering    │
│  - Test thresholds      │
│  - Calculate silhouette │
│  - Select best per      │
│    product              │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Cluster All Data       │
│  - Sample if needed     │
│  - Agglomerative cluster│
│  - KNN for remaining    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Generate Cluster Names │
│  - Sample narratives    │
│  - Create labels        │
│  - Save mappings        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Visualize with UMAP    │
│  - Reduce dimensions    │
│  - Plot by product      │
│  - Save visualizations  │
└─────────────────────────┘
```

---

## 9. Key Design Decisions

### 9.1 Embedding Model Selection
**Decision**: Use `sentence-transformers/all-miniLM-L6-v2`
- **Rationale**: Balance between quality and speed; 384 dimensions manageable for clustering
- **Trade-off**: Smaller model than BERT-base but sufficient for complaint similarity

### 9.2 Clustering Algorithm
**Decision**: Agglomerative clustering with distance thresholds
- **Rationale**: Hierarchical structure allows flexible cluster granularity
- **Alternative considered**: K-means (rejected due to need for predetermined k)

### 9.3 Product-Specific Clustering
**Decision**: Cluster separately by product category
- **Rationale**: Different products have distinct complaint patterns
- **Benefit**: More meaningful clusters within product context

### 9.4 Sampling Strategy
**Decision**: 25,000 sample limit for large datasets
- **Rationale**: Balance between computational efficiency and cluster quality
- **Mitigation**: Use KNN to assign remaining points to established clusters

### 9.5 Narrative Length Filtering
**Decision**: Remove narratives > 4,930 characters (99th percentile)
- **Rationale**: Extreme outliers may affect embedding quality
- **Impact**: Removes <1% of data while improving model stability

---

## 10. Performance Considerations

### 10.1 Bottlenecks
1. **Embedding Generation**: Most time-intensive step; GPU acceleration critical
2. **Clustering Large Datasets**: Mitigated through sampling strategy
3. **UMAP Visualization**: Can be slow for datasets with >50k points

### 10.2 Optimization Strategies
- Batch processing for embeddings (batch_size=32)
- Sampling for clustering threshold optimization
- KNN assignment for remaining points after clustering
- GPU utilization for transformer model

---

## 11. Future Enhancements

### 11.1 Potential Improvements
1. **Dynamic Threshold Selection**: Automate per-product threshold optimization
2. **Incremental Clustering**: Add new complaints without full reprocessing
3. **Multi-level Clustering**: Hierarchical clusters at multiple granularities
4. **Temporal Analysis**: Track cluster evolution over time
5. **Company-Specific Insights**: Cross-company comparison capabilities

### 11.2 Scalability Considerations
- **Distributed Processing**: For datasets exceeding single-machine capacity
- **Online Learning**: Update clusters as new data arrives
- **Model Fine-tuning**: Domain-specific embedding model for financial complaints

---

## 12. Maintenance and Monitoring

### 12.1 Data Quality Checks
- Monitor null narrative rates
- Track product distribution changes
- Validate embedding generation success rates

### 12.2 Model Performance
- Periodic silhouette score evaluation
- Cluster size distribution analysis
- Manual review of cluster coherence

### 12.3 Version Control
- Track optimal threshold changes over time
- Document cluster name evolution
- Maintain embedding model version compatibility

---

## Document Version
- **Version**: 1.0
- **Date**: December 6, 2025
- **Based on**: `load_complaint_data.ipynb`
