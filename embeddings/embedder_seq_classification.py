# Imports
import os
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import evaluate

EMBED_TEXT_SEQ_CLASSIFICATION_DIM = 768 

# Function to assign a unique class to each position in a grid
def position_to_class(row, col, position_to_class_map):
    """
    Assigns a unique class to each position in a grid.
    
    Args:
        row (int): The row position in the grid.
        col (int): The column position in the grid.
        position_to_class_map (dict): Dictionary mapping (row, col) tuples to class indices.
        
    Returns:
        int: The class index for the given position.
    """
    # If the position hasn't been assigned a class yet, assign a new class
    if (row, col) not in position_to_class_map:
        position_to_class_map[(row, col)] = len(position_to_class_map)
    return position_to_class_map[(row, col)]

# Function to preprocess text
def preprocess_text(text):
    """
    Preprocesses text by removing leading and trailing whitespace.
    
    Args:
        text (str): The input text to preprocess.
        
    Returns:
        str: The preprocessed text.
    """
    return text.strip()

# Function to tokenize texts using a BERT tokenizer
def tokenize_function(texts, tokenizer):
    """
    Tokenizes a batch of texts using a BERT tokenizer.
    
    Args:
        texts (list): A list of text strings to tokenize.
        tokenizer (BertTokenizer): The BERT tokenizer.
        
    Returns:
        dict: A dictionary containing tokenized inputs (input_ids, attention_mask, etc.).
    """
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)


# Function to train a BERT model for text classification
def train():
    """
    Trains a BERT model for text classification.
    
    This function:
    1. Loads and preprocesses the dataset
    2. Maps grid positions to class labels
    3. Splits data into train, validation, and test sets
    4. Tokenizes the text data
    5. Trains a BERT classifier
    6. Evaluates the model on the test set
    7. Saves the trained model and tokenizer
    
    Returns:
        None
    """
    # Loading the dataset
    data = pd.read_pickle("dataset\data.pickle")
    data_test = pd.read_pickle("dataset\data_test.pickle")
    
    # Dictionary to keep track of the position-to-class mapping
    position_to_class_map = {}

    # Apply the function with the position-to-class mapping
    data["class"] = data.apply(lambda row: position_to_class(row["row"], row["column"], position_to_class_map), axis=1)
    data["prompt"] = data["prompt"].apply(preprocess_text)
    
    data_test["class"] = data_test.apply(lambda row: position_to_class(row["row"], row["column"], position_to_class_map), axis=1)
    data_test["prompt"] = data_test["prompt"].apply(preprocess_text)

    # Split 'data' into train (80%) and validation (20%)
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        data["prompt"], data["class"], test_size=0.2, random_state=42
    )

    # Use 'data_test' as the test set
    test_texts = data_test["prompt"]
    test_labels = data_test["class"]
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Convert to Hugging Face Dataset
    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()}),
        "valid": Dataset.from_dict({"text": valid_texts.tolist(), "label": valid_labels.tolist()}),
        "test": Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()}),
    })


    # Tokenize datasets
    dataset = dataset.map(lambda x: tokenize_function(x["text"],tokenizer), batched=True)

    # Set format for PyTorch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Load BERT model with classification head
    num_labels = data["class"].nunique()
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    # Freeze all BERT parameters
    # for param in model.bert.parameters():
    #     param.requires_grad = False

    # Define accuracy metric
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        report_to=[],  
        logging_steps=10, 
    )

    # Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Evaluate model
    trainer.evaluate(dataset["test"])
    
    # Save the model and tokenizer
    model.save_pretrained("./models/embedder/seq_classification")
    tokenizer.save_pretrained("./models/embedder/seq_classification")
    
    
# Function to embed text using the trained model
def embed_text_seq_classification(texts, device='cuda'):
    """
    Embeds text using a trained BERT classification model.
    
    This function checks if a trained model exists, trains one if it doesn't,
    and then uses the model to generate embeddings for the input texts.
    
    Args:
        texts (list or str): A single text string or list of text strings to embed.
        device (str): The device to run the model on ('cuda' or 'cpu').
        
    Returns:
        torch.Tensor: The embeddings of the input texts, with shape [batch_size, embedding_dim].
    """
    
    # Load the model and tokenizer
    model_path = "./models/embedder/seq_classification"
    
    # Check if the model exists, else train the model
    if not os.path.exists(model_path):
        print("Model not found. Training a new model for classification...")
        train()
    
    # Load the model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
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
        embeddings = outputs.hidden_states[-1][:, 0, :]  
    
    return embeddings