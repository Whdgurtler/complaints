from transformers import AutoTokenizer, AutoModel
#load models on gpu if available
import torch
import numpy as np

def load_model(model_name: str = "sentence-transformers/all-miniLM-L6-v2" ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model = model.to(device)
    return tokenizer, model 



def embed_text(text):
    #tokenize text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    #get model output
    with torch.no_grad():
        outputs = model(**inputs)
    #mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()
#embed all narratives
def embed_narratives(input_data: pd.DataFrame, batch_size: int = 32, text_column: str = 'consumer_complaint_narrative'):    
    embeddings = []
    batch_size = 32
    for i in range(0, len(input_data), batch_size):
        print(f"Processing batch {i//batch_size + 1}")
        batch_texts = input_data[text_column].iloc[i:i+batch_size].tolist()
        batch_embeddings = embed_text(batch_texts)
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    print(embeddings.shape)
    #save embeddings to pickle file with complaint ids, product, subproduct, company
    embeddings_dict = {
        'complaint_id': input_data['complaint_id'].tolist(),
        'product': input_data['product'].tolist(),
        'sub-product': input_data['sub-product'].tolist(),
        'company': input_data['company'].tolist(),
        'narrative': input_data['consumer_complaint_narrative'].tolist(),
        'embeddings': embeddings
    }
    #write to pickle file
    import pickle
    with open('../data/complaint_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings_dict, f)
    


