import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import re

# ===================== Dataset =====================
class CharDataset(Dataset):
    def __init__(self, data, sequence_length, stride, vocab_size):
        self.data = data
        self.sequence_length = sequence_length
        self.stride = stride
        self.vocab_size = vocab_size
        self.sequences = []
        self.targets = []
        
        # Create overlapping sequences with stride
        for i in range(0, len(data) - sequence_length, stride):
            self.sequences.append(data[i:i + sequence_length])
            self.targets.append(data[i + 1:i + sequence_length + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sequence, target

# ===================== Model =====================
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_dim)
        # TODO: Initialize your model parameters as needed e.g. W_e, W_h, etc.
        # Initialize model parameters
        self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size)) #hidden layer->hidden layer
        self.W_e = nn.Parameter(torch.empty(embedding_dim, hidden_size)) #input->hidden layer
        self.W_o = nn.Parameter(torch.empty(output_size, hidden_size)) #hidden layer->output
        # Apply weight initialization using normal distribution
        nn.init.normal_(self.W_h, mean=0.0, std=0.01)
        nn.init.normal_(self.W_e, mean=0.0, std=0.01)
        nn.init.normal_(self.W_o, mean=0.0, std=0.01)

    def forward(self, x, hidden):
        """
        x in [b, l] # b is batch_size and l is sequence length
        """
        x_embed = self.embedding(x)  # [b=batch_size, l=sequence_length, e=embedding_dim]
        b, l, _ = x_embed.size()
        x_embed = x_embed.transpose(0, 1) # [l, b, e]
        if hidden is None:
            h_t_minus_1 = self.init_hidden(b)
        else:
            h_t_minus_1 = hidden
        output = []
        for t in range(l):
            # RNN forward step for timestamp t
            x_t = x_embed[t]  # [b, e]
            h_t = torch.tanh(x_t @ self.W_e.t() + h_t_minus_1 @ self.W_h.t())  # [b, h]
            output.append(h_t)  # Store the current hidden state
            h_t_minus_1 = h_t  # Update the previous hidden state
        output = torch.stack(output) # [l, b, e]
        output = output.transpose(0, 1) # [b, l, e]
        
        # TODO set these values after completing the loop above
        final_hidden = h_t_minus_1.clone() # [b, h] (clone of final hidden state)
        logits = output @ self.W_o.t()  # Project hidden states to vocab space [b, l, vocab_size=v] 
        return logits, final_hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

# ===================== Training =====================
device = 'cpu'
print(f"Using device: {device}")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z.,!?;:()\[\] ]+', '', text)
    return text

# To debug your model you should start with a simple sequence an RNN should predict this perfectly
sequence = "abcdefghijklmnopqrstuvwxyz" * 100
sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}  # Mapping from character to index
idx_to_char = {idx: char for idx, char in enumerate(vocab)}  # Mapping from index to character
data = [char_to_idx[char] for char in sequence]

#Understand and adjust the hyperparameters once the code is running
sequence_length = 200 # Length of each input sequence
stride = 5            # Stride for creating sequences
embedding_dim = 200      # Dimension of character embeddings
hidden_size = 200        # Number of features in the hidden state of the RNN
learning_rate = 0.05    # Learning rate for the optimizer
num_epochs = 1       # Number of epochs to train
batch_size = 64        # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size, embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
#TODO: Split the data into 90:10 ratio with PyTorch indexing
test_size = len(data_tensor) - train_size # remaining 10%

# Split the dataset
train_data = data_tensor[:train_size] # First 90% for training
test_data = data_tensor[train_size:] # Last 10% for testing

train_dataset = CharDataset(train_data, sequence_length, stride, vocab_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    hidden = None
    for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        output, hidden = model(batch_inputs, hidden)
        # Detach removes the hidden state from the computational graph.
        # This is prevents backpropagating through the full history, and
        # is important for stability in training RNNs
        hidden = hidden.detach()

        # TODO compute the loss, backpropagate gradients, and update total_loss
        # Reshape logits and targets for the loss computation
        logits = output.view(-1, output_size)  # (batch_size * sequence_length, output_size)
        batch_targets = batch_targets.view(-1)  # (batch_size * sequence_length)

        # Compute loss
        loss = criterion(logits, batch_targets)

        # Backpropagation and optimization
        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Backpropagate loss
        optimizer.step()  # Optimize parameters
        total_loss += loss.item()  # Accumulate loss


    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
# ===================== Test Loop =====================
test_dataset = CharDataset(test_data, sequence_length, stride, vocab_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

total_test_loss = 0
num_batches = 0  # To calculate average loss
model.eval()  # Set model to evaluation mode.

with torch.no_grad():  # Disable gradient computation
    hidden = None  # Initialize hidden state
    for batch_inputs, batch_targets in tqdm(test_loader, desc="Testing:"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        hidden = model.init_hidden(batch_inputs.size(0)).to(device)  # Initialize hidden state to match batch_size
        # Get model outputs
        output, hidden = model(batch_inputs, hidden)
        hidden = hidden.detach()  # Detach to prevent computing gradients through the entire history
        
        # Reshape logits and targets for loss computation
        logits = output.view(-1, output_size)  # (batch_size * sequence_length, output_size)
        batch_targets = batch_targets.view(-1)  # (batch_size * sequence_length)

        # Compute test loss
        loss = criterion(logits, batch_targets)

        total_test_loss += loss.item()  # Accumulate loss
        num_batches += 1  # Increment batch count

# Calculate average loss
average_test_loss = total_test_loss / num_batches if num_batches > 0 else float('inf')
print(f"Average Test Loss: {average_test_loss:.4f}")

# ===================== Text Generation =====================
def sample_from_output(logits, temperature=1.0):
    """
    Sample from the logits with temperature scaling.
    logits: Tensor of shape [batch_size, vocab_size] (raw scores, before softmax)
    temperature: a float controlling the randomness (higher = more random)
    """
    if temperature <= 0:
        temperature = 0.00000001
    # Apply temperature scaling to logits (increase randomness with higher values)
    scaled_logits = logits / temperature 
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(scaled_logits, dim=1)
    
    # Sample from the probability distribution
    sampled_idx = torch.multinomial(probabilities, 1)
    return sampled_idx

def generate_text(model, start_text, n, k, temperature=1.0):
    """
        model: The trained RNN model used for character prediction.
        start_text: The initial string of length `n` provided by the user to start the generation.
        n: The length of the initial input sequence.
        k: The number of additional characters to generate.
        temperature: Optional
        A scaling factor for randomness in predictions. Higher values (e.g., >1) make 
            predictions more random, while lower values (e.g., <1) make predictions more deterministic.
            Default is 1.0.
    """
    start_text = start_text.lower()
    model.eval()  # Set the model to evaluation mode
    generated_text = start_text

    # Convert start_text to tensor indices
    input_indices = torch.tensor([char_to_idx[char] for char in start_text]).unsqueeze(0).to(device)
    hidden = None  # Initialize the hidden state

    for _ in range(k):
        # Get the model's output
        output, hidden = model(input_indices, hidden)
        
        # Get the logits for the last character prediction
        logits = output[:, -1, :]  # Last timestep's predictions
        
        # Sample from the output probabilities
        next_index = sample_from_output(logits, temperature).item()  # Get the next character index
        
        # Append the predicted character to the generated text
        generated_text += idx_to_char[next_index]
        
        # Update the input for the next iteration
        input_indices = torch.cat((input_indices, torch.tensor([[next_index]]).to(device)), dim=1)

    return generated_text

print("Training complete. Now you can generate text.")
while True:
    start_text = input("Enter the initial text (n characters, or 'exit' to quit): ")
    
    if start_text.lower() == 'exit':
        print("Exiting...")
        break
    
    n = len(start_text) 
    k = int(input("Enter the number of characters to generate: "))
    temperature_input = input("Enter the temperature value (1.0 is default, >1 is more random): ")
    temperature = float(temperature_input) if temperature_input else 1.0
    
    completed_text = generate_text(model, start_text, n, k, temperature)
    
    print(f"Generated text: {completed_text}")