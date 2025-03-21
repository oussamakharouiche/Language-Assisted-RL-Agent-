from transformers import BertTokenizer, BertModel
import torch

EMBED_TEXT_DIM = 768 

def embed_text(texts, device='cuda'):
    """
    Embeds text using a BERT.
    
    Args:
        texts (list or str): A single text string or list of text strings to embed.
        device (str): The device to run the model on ('cuda' or 'cpu').
        
    Returns:
        torch.Tensor: The embeddings of the input texts.
    """
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)
    
    # Tokenize the input texts
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    # Move the input tensors to the correct device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Get embeddings from the model 
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Get the embeddings from the last layer (CLS token)
        embeddings = outputs.last_hidden_state[:, 0, :]  
    
    return embeddings