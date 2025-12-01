def sample_narratives_by_product_cluster(df, product, cluster_col, n_samples=10):       
    """
    Sample narratives for a given product and cluster.

    Args:
        df (pd.DataFrame): DataFrame containing complaint data.
        product (str): The product to filter by.
        cluster_col (str): The column name for cluster labels.
        n_samples (int): Number of samples to return per cluster.
        
    Returns:
        list: List of sampled narratives.
    """
    product_df = df[df['product'] == product]
    sampled_narratives = []
    for cluster in product_df[cluster_col].unique():
        cluster_df = product_df[product_df[cluster_col] == cluster]
        sampled_narratives.extend(
            cluster_df['consumer_complaint_narrative'].dropna().sample(
                n=min(n_samples, len(cluster_df)), random_state=42
            ).tolist()
        )
    return sampled_narratives
def create_cluster_name(sample_narratives, model=None, client=None):
    """
    Create a descriptive name for a cluster based on sample narratives using huggingface model.

    Args:
        sample_narratives (list): List of sample narratives from the cluster.
        model (str): The Hugging Face model to use (default: "Qwen/Qwen2.5-72B-Instruct").
        client (InferenceClient): Optional pre-initialized client for reuse.

    Returns:
        str: Descriptive name for the cluster.
    """
    from huggingface_hub import InferenceClient
    
    if model is None:
        model = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    
    # Combine all narratives without truncation to preserve full context
    combined_text = "\n---\n".join(sample_narratives)
    
    # Create prompt for naming the cluster
    user_prompt = f"""Analyze these customer complaints and identify the main theme.

{combined_text}

Respond with only a concise category name or phrase that describes the main issue:"""
    
    # Check prompt length and truncate if needed (rough token estimate: 1 token ≈ 4 chars)
    max_chars = 12000  # Conservative limit for most models (~3000 tokens)
    if len(user_prompt) > max_chars:
        # Reduce narratives until prompt fits
        truncated_narratives = sample_narratives[:5]  # Start with 5
        combined_text = "\n---\n".join(truncated_narratives)
        user_prompt = f"""Read these customer complaints and respond with ONLY a 2-4 word category name:

{combined_text}

Category:"""
        
        # If still too long, use even fewer
        if len(user_prompt) > max_chars:
            truncated_narratives = sample_narratives[:3]
            combined_text = "\n---\n".join(truncated_narratives)
            user_prompt = f"""Read these customer complaints and respond with ONLY a 2-4 word category name:

{combined_text}

Category:"""
    
    try:
        # Reuse client if provided, otherwise create new one
        if client is None:
            client = InferenceClient(model=model)
        
        # Use conversational API
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        response = client.chat_completion(
            messages=messages,
            max_tokens=20,
            temperature=0.7
        )
        
        # Extract the response text
        message = response.choices[0].message
        # GPT-OSS-20B uses reasoning field, try both content and reasoning
        if hasattr(message, 'content') and message.content:
            cluster_name = message.content.strip()
        elif hasattr(message, 'reasoning') and message.reasoning:
            # Extract actual answer from reasoning field
            reasoning_text = message.reasoning.strip()
            # The model often includes the task in reasoning, extract just the category name
            # Look for patterns like "Category: X" or just use the last meaningful phrase
            if ':' in reasoning_text:
                cluster_name = reasoning_text.split(':')[-1].strip()
            else:
                # Take last sentence or phrase as the answer
                cluster_name = reasoning_text.split('.')[-1].strip()
        else:
            cluster_name = ""
        
        # Clean up and take first line only
        cluster_name = cluster_name.split('\n')[0]
        # Remove quotes if present
        cluster_name = cluster_name.strip('"\'')
        
        return cluster_name if cluster_name else "Unnamed Cluster"
    except Exception as e:
        import traceback
        print(f"Error generating cluster name:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return "Unnamed Cluster"

def create_cluster_names_batch(df, product, cluster_col, n_samples=10, model=None):
    """
    Efficiently generate names for all clusters in a product using a single client.
    
    Args:
        df (pd.DataFrame): DataFrame containing complaint data.
        product (str): The product to filter by.
        cluster_col (str): The column name for cluster labels.
        n_samples (int): Number of samples per cluster (default: 5).
        model (str): The Hugging Face model to use.
        
    Returns:
        dict: Mapping of cluster_id to cluster_name.
    """
    from huggingface_hub import InferenceClient
    
    if model is None:
        model = "openai/gpt-oss-20b"
    
    # Initialize client once for all requests
    client = InferenceClient(model=model)
    
    product_df = df[df['product'] == product]
    cluster_names = {}
    
    for cluster_id in product_df[cluster_col].unique():
        cluster_df = product_df[product_df[cluster_col] == cluster_id]
        sample_narratives = cluster_df['consumer_complaint_narrative'].dropna().sample(
            n=min(n_samples, len(cluster_df)), random_state=42
        ).tolist()
        
        cluster_name = create_cluster_name(sample_narratives, model=model, client=client)
        cluster_names[cluster_id] = cluster_name
        
    return cluster_names