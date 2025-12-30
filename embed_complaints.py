import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel


class ComplaintEmbedder:
    """Generate embeddings for complaint narratives using transformer models."""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a specific model.
        
        Args:
            model_name (str): HuggingFace model identifier
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def embed_text(self, texts):
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts (list): List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings
        """
        inputs = self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
    
    def embed_dataframe(self, df, text_column='consumer_complaint_narrative', batch_size=32):
        """
        Generate embeddings for all texts in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Name of column containing text to embed
            batch_size (int): Batch size for processing
            
        Returns:
            tuple: (embeddings_array, embeddings_df) containing embeddings and metadata
        """
        embeddings = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_num = i // batch_size + 1
            print(f"Processing batch {batch_num}/{total_batches}")
            
            batch_texts = df[text_column].iloc[i:i+batch_size].tolist()
            batch_embeddings = self.embed_text(batch_texts)
            embeddings.append(batch_embeddings)
        
        embeddings_array = np.vstack(embeddings)
        print(f"Generated {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]}")
        
        return embeddings_array
    
    def save_embeddings(self, embeddings_array, df, output_path, 
                       metadata_cols=None):
        """
        Save embeddings with metadata to pickle file.
        
        Args:
            embeddings_array (np.ndarray): Array of embeddings
            df (pd.DataFrame): DataFrame with metadata
            output_path (str): Path to save pickle file
            metadata_cols (list): List of column names to include as metadata
        """
        if metadata_cols is None:
            metadata_cols = ['complaint_id', 'product', 'sub-product', 
                           'company', 'consumer_complaint_narrative']
        
        # Create embeddings DataFrame
        embeddings_df = pd.DataFrame(embeddings_array)
        
        # Add metadata columns
        for col in metadata_cols:
            if col in df.columns:
                embeddings_df[col] = df[col].values
        
        embeddings_df.to_pickle(output_path)
        print(f"Saved embeddings to {output_path}")
        
        return embeddings_df


def generate_embeddings(df, model_name="sentence-transformers/all-MiniLM-L6-v2", 
                       output_path=None, batch_size=32):
    """
    Convenience function to generate embeddings from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with complaint data
        model_name (str): Model to use for embeddings
        output_path (str): Path to save embeddings (optional)
        batch_size (int): Batch size for processing
        
    Returns:
        tuple: (embeddings_array, embeddings_df)
    """
    embedder = ComplaintEmbedder(model_name=model_name)
    embeddings_array = embedder.embed_dataframe(df, batch_size=batch_size)
    
    if output_path:
        embeddings_df = embedder.save_embeddings(embeddings_array, df, output_path)
        return embeddings_array, embeddings_df
    
    return embeddings_array
