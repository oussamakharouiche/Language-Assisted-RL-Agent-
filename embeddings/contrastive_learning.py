import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertTokenizer, BertModel, BertForSequenceClassification
import math
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
import torch.nn.functional as F
import numpy as np

###############################################################################
#                                 Utilities
###############################################################################

def semantic_dist(embedding1, embedding2):
    """
    Computes a simple semantic distance between two embeddings.

    The distance is computed as the sum of squared differences 
    across all dimensions of the embedding.

    Args:
        embedding1 (torch.Tensor): The first embedding.
        embedding2 (torch.Tensor): The second embedding.
    
    Returns:
        torch.Tensor: A scalar tensor representing the distance.
    """
    # Difference between the two embeddings
    distance_embedding = (embedding1 - embedding2)
    # Sum of squared differences
    distance = torch.sum(distance_embedding ** 2, dim=0)
    return distance


def margin(pos1, pos2, embedding1, embedding2, alpha=1.0, beta=0.5):
    """
    Computes a combined margin that depends on:
      1) The semantic distance between the two embeddings.
      2) The 'true' distance (positional difference) between pos1 and pos2.

    margin = alpha * semantic_distance + beta * true_distance

    Args:
        pos1 (torch.Tensor): 2D tensor representing the position of the first element (e.g., row/column).
        pos2 (torch.Tensor): 2D tensor representing the position of the second element.
        embedding1 (torch.Tensor): The first embedding.
        embedding2 (torch.Tensor): The second embedding.
        alpha (float): Weight for the semantic distance term.
        beta (float): Weight for the positional distance term.

    Returns:
        torch.Tensor: A scalar margin for each pair.
    """
    # Compute the semantic distance using the function above
    semantic_distance = semantic_dist(embedding1, embedding2)

    # Compute the 'true' distance between positions
    # We assume pos1, pos2 are shape (N, 2), so sum over axis=1
    true_distance = torch.sum((pos1 - pos2) ** 2, axis=1)

    # Weighted sum of these distances
    combined_margin = alpha * semantic_distance + beta * true_distance
    return combined_margin


def criterion(queries, keys, margins, labels):
    """
    Custom loss function that combines positive and negative losses.

    For positive pairs (label=1):
        loss = sum((query - key)^2)
    
    For negative pairs (label=0):
        loss = ReLU(margin - (query - key)^2)

    Then we take the mean of all individual pair losses.

    Args:
        queries (torch.Tensor): Batch of transformed query embeddings.
        keys (torch.Tensor): Batch of transformed key embeddings.
        margins (torch.Tensor): The margin values associated with each pair.
        labels (torch.Tensor): Binary labels (1 for positive pair, 0 for negative pair).

    Returns:
        torch.Tensor: A scalar representing the average loss over the batch.
    """
    # Positive pair loss: sum of squared differences
    pos_loss = torch.sum((queries - keys) ** 2, dim=1)

    # Negative pair loss: ReLU(margins - positive_loss_for_that_pair)
    # We multiply the positive loss by (1 - labels) so that it only applies to negative pairs
    neg_loss = torch.relu(margins - pos_loss * (1 - labels))

    # Weighted combination: label * pos_loss + (1 - label) * neg_loss
    total_loss = labels * pos_loss + (1 - labels) * neg_loss

    # Return the average loss across the batch
    return torch.mean(total_loss)

###############################################################################
#                              Model Definition
###############################################################################

class EmbeddingTransformer(nn.Module):
    """
    A Transformer-based module to project (768-dim BERT embeddings) 
    into a new embedding space and then project back to the original dimension.

    Args:
        input_dim (int): Dimensionality of input embeddings.
        hidden_dim (int): Hidden dimensionality within the Transformer layers.
        num_layers (int): Number of Transformer encoder layers.
        num_heads (int): Number of attention heads in each Transformer layer.
    """
    def __init__(self, input_dim=20, hidden_dim=64, num_layers=3, num_heads=4):
        super().__init__()

        # Linear layer to project input embeddings to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Define a single encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        # Stack the encoder layer multiple times
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear layer to project back to the original input dimension
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        """
        Forward pass of the embedding transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len=1, input_dim). 
                              Often used with shape (batch_size, input_dim) if we treat 
                              the sequence length as 1.

        Returns:
            torch.Tensor: Output tensor of the same shape as `x` (batch_size, input_dim).
        """
        # Project input embeddings to hidden_dim
        x = self.input_proj(x)

        # If input is (batch_size, dim), make it (batch_size, 1, dim) to match the transformer's expected shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # shape: (batch_size, 1, hidden_dim)

        # Apply the Transformer encoder
        x = self.transformer(x)  # shape: (batch_size, 1, hidden_dim)

        # Remove the sequence dimension
        x = x.squeeze(1)  # shape: (batch_size, hidden_dim)

        # Project back to the original dimension
        x = self.output_proj(x)  # shape: (batch_size, input_dim)
        return x

###############################################################################
#                            Dataset Preparation
###############################################################################

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


def prepare_dataset(dataframe, tokenizer, bert_model, n_pos_pair=1, n_neg_pair=1):
    """
    Prepares a dataset of positive and negative pairs from a given DataFrame.

    1) It looks for corners in the data (positions like (0,9), (9,0), (9,9) by default).
    2) For each row in the corners, it randomly samples:
       - n_pos_pair positive pairs from the same corner
       - n_neg_pair negative pairs from outside the corner
    3) Creates pairs with the text, positions, and BERT embeddings.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing columns ['row', 'column', 'prompt'].
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to process text.
        bert_model (transformers.PreTrainedModel): The BERT model to generate embeddings.
        n_pos_pair (int): Number of positive pairs to sample per corner sample.
        n_neg_pair (int): Number of negative pairs to sample per corner sample.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            "query", "key", "embedding1", "embedding2", "label", "margin", etc.
    """
    samples = []

    # Define corner positions to look for
    corners = [(0, 9), (9, 0), (9, 9)]

    # Iterate over each corner
    for x, y in corners:
        # Filter the dataframe to find rows that match the corner
        df_corner = dataframe[(dataframe.row == x) & (dataframe.column == y)]
        
        # If no rows found for this corner, skip
        if df_corner.empty:
            continue 

        # Indices outside the current corner
        non_corner_idx = dataframe.index.difference(df_corner.index)

        # For every row in the corner
        for _, row in df_corner.iterrows():
            # Current row's position
            pos1_tensor = torch.tensor([row["row"], row["column"]], dtype=torch.float).unsqueeze(0)
            query_text = row["prompt"]

            # === Positive Pairs (from same corner) ===
            df_sampled = df_corner.sample(n=min(n_pos_pair, len(df_corner) - 1), replace=False)
            pos_pairs = []
            for _, sample_row in df_sampled.iterrows():
                pos_pairs.append({
                    "query": preprocess_text(query_text),
                    "key": preprocess_text(sample_row["prompt"]),
                    "embedding1": None,
                    "embedding2": None,
                    "pos1_tensor": pos1_tensor,
                    "pos2_tensor": torch.tensor([sample_row["row"], sample_row["column"]], dtype=torch.float).unsqueeze(0),
                    "label": 1
                })

            # === Negative Pairs (from outside the corner) ===
            df_left = dataframe.loc[non_corner_idx].sample(n=min(n_neg_pair, len(non_corner_idx)), replace=False)
            neg_pairs = []
            for _, sample_row in df_left.iterrows():
                neg_pairs.append({
                    "query": preprocess_text(query_text),
                    "key": preprocess_text(sample_row["prompt"]),
                    "embedding1": None,
                    "embedding2": None,
                    "pos1_tensor": pos1_tensor,
                    "pos2_tensor": torch.tensor([sample_row["row"], sample_row["column"]], dtype=torch.float).unsqueeze(0),
                    "label": 0
                })

            # Combine positive and negative pairs
            all_pairs = pos_pairs + neg_pairs

            # Prepare texts for batch tokenization
            queries = [pair["query"] for pair in all_pairs]
            keys = [pair["key"] for pair in all_pairs]

            # Tokenize
            inputs1 = tokenizer(queries, padding=True, truncation=True, return_tensors='pt')
            inputs2 = tokenizer(keys, padding=True, truncation=True, return_tensors='pt')

            # Generate BERT embeddings
            with torch.no_grad():
                outputs1 = bert_model(**inputs1).last_hidden_state[:, 0, :]  # [CLS] token's embeddings
                outputs2 = bert_model(**inputs2).last_hidden_state[:, 0, :]

            # Assign computed embeddings and margins
            for i, pair in enumerate(all_pairs):
                pair["embedding1"] = outputs1[i]
                pair["embedding2"] = outputs2[i]

                # Compute the margin for each pair
                pair["margin"] = margin(
                    pair["pos1_tensor"], 
                    pair["pos2_tensor"], 
                    outputs1[i], 
                    outputs2[i]
                )
                
                # Remove position tensors (no longer needed after margin calculation)
                del pair["pos1_tensor"], pair["pos2_tensor"]
            
            # Accumulate all pairs
            samples.extend(all_pairs)

    return samples


###############################################################################
#                              Training Routine
###############################################################################

def train(model, train_set, batch_size=64):
    """
    Trains the given model on the provided training set.

    1) Creates a DataLoader from train_set.
    2) Uses AdamW optimizer with specified hyperparameters.
    3) Runs a training loop for a fixed number of epochs and logs the loss.

    Args:
        model (nn.Module): The model to train.
        train_set (list[dict]): The training dataset (list of samples).
        batch_size (int): Batch size to use in DataLoader.

    Returns:
        None. (Displays the training loss and a plot of loss over epochs.)
    """
    # Device selection (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # DataLoader for the training set
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)

    # Training loop parameters
    num_epochs = 200
    num_epochs_display = num_epochs // 10
    losses = []

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        total_loss = 0.0

        for index, batch in enumerate(train_loader, start=1):
            # Retrieve input embeddings and other data
            queries = batch["embedding1"].to(device)
            keys = batch["embedding2"].to(device)
            labels = batch["label"].to(device)
            margins = batch["margin"].to(device)

            # Forward pass: transform queries and keys
            output_1 = model(queries)
            output_2 = model(keys)

            # Compute custom loss
            loss = criterion(output_1, output_2, margins, labels)

            # Accumulate loss
            total_loss += loss.item()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Average loss for the epoch
        total_loss /= index
        losses.append(total_loss)
        
        # Periodic logging
        if epoch % num_epochs_display == 0 or epoch == num_epochs - 1:
            tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    os.makedirs('./saved_model', exist_ok=True)

    # Save only the model parameters 
    torch.save(model.state_dict(), './saved_model/model_cl.pt')

    # Plot training loss
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1, len(losses) + 1), losses, marker="o", linestyle="-")
    plt.show()

###############################################################################
#                                Evaluation
###############################################################################

def plot_figure(model, test_set, device='cpu', n_samples=20):
    """
    Samples N pairs from the test_set and compares:
      1) Cosine similarity of the new model's embeddings
      2) Cosine similarity of the original embeddings (baseline)

    Args:
        model (nn.Module): The trained model to evaluate.
        test_set (list[dict]): The list of sample dictionaries for testing.
        device (str): The device to run on ('cuda' or 'cpu').
        n_samples (int): Number of random samples to evaluate.

    Returns:
        tuple: (results, results_base) where each is a list of tuples 
               (cosine_similarity, label).
    """
    model.eval()
    results = []
    results_base = []

    # Choose random indices from the test_set
    indices = np.random.choice(len(test_set), size=n_samples, replace=False)

    # Evaluate each chosen sample
    for index in indices:
        sample = test_set[index]

        # Move baseline embeddings to device
        key_baseline = sample["embedding1"].to(device)
        query_baseline = sample["embedding2"].to(device)

        # Apply the model to get the "transformed" embeddings
        key_embedding = model(key_baseline.unsqueeze(0))
        query_embedding = model(query_baseline.unsqueeze(0))

        # Compute new model's cosine similarity
        cos_similarity_new = F.cosine_similarity(key_embedding, query_embedding)

        # Compute baseline cosine similarity
        cos_similarity_base = F.cosine_similarity(
            key_baseline.unsqueeze(0), query_baseline.unsqueeze(0)
        )

        # Store results along with labels
        results.append((cos_similarity_new, sample["label"]))
        results_base.append((cos_similarity_base, sample["label"]))

    return results, results_base


def plot_results(results, results_base):
    """
    Plots two histograms:
      1) Cosine similarities computed by the new model.
      2) Baseline cosine similarities (original BERT embeddings).

    Args:
        results (list[tuple]): List of tuples (similarity_tensor, label).
        results_base (list[tuple]): List of tuples (similarity_tensor, label).

    Returns:
        None. (Displays two histograms.)
    """
    # Extract numeric similarity scores
    cos_scores = [float(item[0]) for item in results]
    cos_scores_base = [float(item[0]) for item in results_base]

    # === First plot: distribution of new model's similarities ===
    plt.figure()
    plt.hist(cos_scores)
    plt.title("Distribution of Cosine Similarities (New Model)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.show()

    # === Second plot: distribution of baseline similarities ===
    plt.figure()
    plt.hist(cos_scores_base)
    plt.title("Distribution of Cosine Similarities (Baseline)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.show()


def gather_model_embeddings(test_set, model, device='cpu'):
    """
    Gathers the model-transformed embeddings and labels for each sample in the test set.

    Args:
        test_set (list[dict]): The test dataset (list of samples).
        model (nn.Module): The trained model that transforms embeddings.
        device (str): The device to run on ('cuda' or 'cpu').

    Returns:
        tuple: (embeddings, labels) where
            embeddings is a (N, D) tensor of model-transformed embeddings
            labels is a (N,) tensor of labels (0 or 1).
    """
    all_model_embeddings = []
    all_labels = []
    
    model.eval()
    for sample in test_set:
        # Baseline embedding
        baseline_embedding = sample["embedding1"].to(device)

        # Model-transformed embedding
        with torch.no_grad():
            model_embedding = model(baseline_embedding.unsqueeze(0))  # shape: (1, D)
        model_embedding = model_embedding.squeeze(0).cpu()            # shape: (D,)

        # Accumulate
        all_model_embeddings.append(model_embedding)
        all_labels.append(sample["label"])
    
    embeddings = torch.stack(all_model_embeddings, dim=0)  # (N, D)
    labels = torch.tensor(all_labels, dtype=torch.long)    # (N,)
    return embeddings, labels


def gather_baseline_embeddings(test_set):
    """
    Gathers the original (baseline) BERT embeddings and labels for each sample in the test set.

    Args:
        test_set (list[dict]): The test dataset (list of samples).

    Returns:
        tuple: (embeddings, labels) where
            embeddings is a (N, D) tensor of baseline embeddings
            labels is a (N,) tensor of labels (0 or 1).
    """
    all_baseline_embeddings = []
    all_labels = []

    for sample in test_set:
        baseline_embedding = sample["embedding1"]  # shape: (D,)
        all_baseline_embeddings.append(baseline_embedding)
        all_labels.append(sample["label"])

    embeddings = torch.stack(all_baseline_embeddings, dim=0)  # (N, D)
    labels = torch.tensor(all_labels, dtype=torch.long)       # (N,)
    return embeddings, labels


def compute_mean_similarity_matrix(embeddings, labels):
    """
    Computes a 2x2 matrix of average cosine similarities:
      M[neg,neg], M[neg,pos], M[pos,neg], M[pos,pos]

    Steps:
      1) Normalize embeddings to unit norm.
      2) Compute the NxN similarity matrix.
      3) Accumulate sums and counts for each (label_i, label_j) cell.
      4) Compute means.

    Args:
        embeddings (torch.Tensor): (N, D) of embeddings.
        labels (torch.Tensor): (N,) containing 0 (neg) or 1 (pos).

    Returns:
        torch.Tensor: 2x2 matrix of average similarities.
    """
    # 1) Normalize to get cos-sim via dot product
    normed = F.normalize(embeddings, p=2, dim=1)  # (N, D)
    sim_matrix = normed @ normed.T                # (N, N)

    # 2) Accumulate sums and counts
    sums = torch.zeros(2, 2, dtype=torch.float32)
    counts = torch.zeros(2, 2, dtype=torch.int32)

    N = labels.shape[0]
    for i in range(N):
        for j in range(N):
            lbl_i = labels[i].item()  # 0 or 1
            lbl_j = labels[j].item()  # 0 or 1

            sim = sim_matrix[i, j].item()
            sums[lbl_i, lbl_j] += sim
            counts[lbl_i, lbl_j] += 1

    # 3) Compute means
    means = torch.zeros(2, 2, dtype=torch.float32)
    for r in [0, 1]:
        for c in [0, 1]:
            if counts[r, c] > 0:
                means[r, c] = sums[r, c] / counts[r, c]
            else:
                means[r, c] = 0.0  
    return means


def plot_2x2_heatmap(sim_matrix, title):
    """
    Plots a single 2x2 heatmap from the given similarity matrix.

    Args:
        sim_matrix (torch.Tensor): 2x2 matrix of similarities.
        title (str): Title for the heatmap.

    Returns:
        None. (Displays a heatmap with colorbar.)
    """
    plt.figure()
    plt.imshow(sim_matrix, aspect='equal')  
    plt.title(title)

    # Print numeric values in each cell
    for row in range(2):
        for col in range(2):
            value = sim_matrix[row, col].item()
            plt.text(col, row, f"{value:.3f}", ha='center', va='center')

    plt.xticks([0, 1], ["neg", "pos"])
    plt.yticks([0, 1], ["neg", "pos"])
    plt.colorbar()
    plt.show()


def plot_pos_neg_heatmaps(test_set, model, base=False, device='cpu'):
    """
    Plots 2x2 average cosine similarity matrices for model-based and baseline embeddings.

    1) Gathers model embeddings & baseline embeddings from test_set.
    2) Computes 2x2 average similarity matrices for each.
    3) Plots them as heatmaps.

    Args:
        test_set (list[dict]): The test dataset (list of samples).
        model (nn.Module): The trained model to transform embeddings.
        base (bool): If True, plot only the baseline matrix. If False, plot only the model's matrix.
        device (str): The device to run on ('cuda' or 'cpu').

    Returns:
        None. (Displays one or two heatmaps depending on the 'base' flag.)
    """
    # 1) Gather model-based embeddings
    model_embeddings, labels_model = gather_model_embeddings(test_set, model, device=device)

    # 2) Gather baseline embeddings
    baseline_embeddings, labels_baseline = gather_baseline_embeddings(test_set)

    # 3) Compute 2×2 average similarity matrices
    mean_sim_model = compute_mean_similarity_matrix(model_embeddings, labels_model)
    mean_sim_baseline = compute_mean_similarity_matrix(baseline_embeddings, labels_baseline)

    # 4) Plot each matrix in its own figure
    if not base:
        plot_2x2_heatmap(mean_sim_model, title="Avg Cosine Similarity (New Model)")
    else:
        plot_2x2_heatmap(mean_sim_baseline, title="Avg Cosine Similarity (Baseline)")


###############################################################################
#                              Embedding Function
###############################################################################

def embed_text(texts, device='cuda'):
    """
    Embeds text using a (potentially) trained BERT classification model.

    This function:
      1) Checks if a saved model exists at './saved_model'. If not, it assumes 
         you will train a classification model and save it there (not covered here).
      2) Loads the saved model and tokenizer.
      3) Tokenizes the input text(s).
      4) Returns the final CLS-token embeddings from BERT.

    Args:
        texts (str or list[str]): A single string or a list of strings to embed.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: A tensor of shape [batch_size, embedding_dim] containing 
                      the CLS embeddings for each input text.
    """
    # Ensure texts is a list for batching
    if isinstance(texts, str):
        texts = [texts]

    model_path = "./saved_model"

    # If the directory doesn't exist, you could train a classification model here
    # For now, just raise an error or handle as needed
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Saved model not found at './saved_model'. Please train and save a model first."
        )

    # Model's Hyperparameters
    input_dim = 768
    hidden_dim = 1024
    num_layers = 3
    num_heads = 4

    # Load your classification model and tokenizer
    model = EmbeddingTransformer(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        num_heads=num_heads
         ).to(device)

    # Complete the weight loading
    checkpoint_path = os.path.join(model_path, "model_cl.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    model.eval()

    # Tokenize Text
    texts_processed = preprocess_text(texts)
    inputs = tokenizer(texts_processed, padding=True, truncation=True, return_tensors='pt')
    outputs = bert_model(**inputs).last_hidden_state[:, 0, :]

    # Compute Embedding
    embeddings = model(outputs)


    return embeddings


###############################################################################
#                                   Main
###############################################################################

if __name__ == "__main__":
    # Prepare your BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Load data
    df = pd.read_pickle("dataset/data.pickle")       # Adjust path
    df_test = pd.read_pickle("dataset/data_test.pickle")  # Adjust path

    # Split data for training/validation
    df_train = df.sample(frac=0.8, random_state=42)
    df_val = df.drop(df_train.index)

    # Prepare the dataset of pairs
    dataset = prepare_dataset(df_train, tokenizer, bert_model)

    # Split dataset further into train/val/test
    train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=42)
    test_set, val_set = train_test_split(test_set, test_size=0.5, random_state=42)

    # Define model parameters for the embedding transformer
    input_dim = 768
    hidden_dim = 1024
    num_layers = 3
    num_heads = 4

    # Create the embedding transformer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingTransformer(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        num_heads=num_heads
    ).to(device)

    # Train the model
    train(model, train_set)

    # Evaluate on test set
    results, results_base = plot_figure(model, test_set, device=device, n_samples=20)
    plot_results(results, results_base)

    # Visualize pos/neg heatmaps
    plot_pos_neg_heatmaps(test_set, model, device=device)        # new model
    plot_pos_neg_heatmaps(test_set, model, base=True, device=device)  # baseline
