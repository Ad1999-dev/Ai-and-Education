import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        # Initialize model parameters
        self.W_ih = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_ho = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.b_o = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, hidden=None):
        """
        x in [b, l] # b is batch_size and l is sequence length
        """
        # Embed the input character indices
        x_embed = self.embedding(x)  # Shape: [b, l, embedding_dim]
        b, l, e = x_embed.size()
        x_embed = x_embed.transpose(0, 1)  # Shape: [l, b, embedding_dim]

        if hidden is None:
            h_t_minus_1 = torch.zeros(b, self.hidden_size, device=x.device)
        else:
            h_t_minus_1 = hidden

        output = []  # List to store hidden states for each time step
        for t in range(l):
            # Calculate the new hidden state using the RNN equations
            h_t = torch.tanh(x_embed[t] @ self.W_ih + h_t_minus_1 @ self.W_hh + self.b_h)
            output.append(h_t)
            h_t_minus_1 = h_t

        output = torch.stack(output)  # Stack to get shape [l, b, hidden_size]
        output = output.transpose(0, 1)  # Transpose to get shape [b, l, hidden_size]

        # Compute logits: the predicted output for each time step using the hidden states
        logits = output @ self.W_ho + self.b_o  # Shape: [b, l, output_size]

        final_hidden = h_t.clone()  # Final hidden state after processing the entire input sequence

        return logits, final_hidden

# ===================== Training =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z.,!?;:()\[\] ]+', '', text)
    return text

# To debug your model you should start with a simple sequence an RNN should predict this perfectly
#sequence = "abcdefghijklmnopqrstuvwxyz" * 100
sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}
data = [char_to_idx[char] for char in sequence]

# Hyperparameters
sequence_length = 10    # Length of each input sequence
stride = 1              # Stride for creating sequences
embedding_dim = 16      # Dimension of character embeddings
hidden_size = 32        # Number of features in the hidden state of the RNN
learning_rate = 0.01    # Learning rate for the optimizer
num_epochs = 5          # Number of epochs to train
batch_size = 32         # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size, embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]

train_dataset = CharDataset(train_data, sequence_length, stride, output_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = CharDataset(test_data, sequence_length, stride, output_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for i, (batch_inputs, batch_targets) in enumerate(train_loader):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output, _ = model(batch_inputs)
        
        # Reshape output and targets for loss calculation
        b, l, v = output.size()
        output_flat = output.reshape(b * l, v)
        targets_flat = batch_targets.reshape(b * l)
        
        # Compute loss
        loss = criterion(output_flat, targets_flat)
        
        # Backpropagation
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update total loss
        total_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}", end='\r')
    
    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# Test the model
model.eval()
total_test_loss = 0
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for batch_inputs, batch_targets in test_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        
        # Forward pass
        output, _ = model(batch_inputs)
        
        # Reshape for loss calculation
        b, l, v = output.size()
        output_flat = output.reshape(b * l, v)
        targets_flat = batch_targets.reshape(b * l)
        
        # Compute loss
        loss = criterion(output_flat, targets_flat)
        total_test_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(output_flat, 1)
        total_predictions += targets_flat.size(0)
        correct_predictions += (predicted == targets_flat).sum().item()

if len(test_loader) > 0:
    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = 100 * correct_predictions / total_predictions
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")
else:
    print("Warning: Test loader is empty. Check your test data and parameters.")

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
    model.eval()  # Set the model to evaluation mode
    start_text = start_text.lower()
    
    # Convert start_text to indices
    input_indices = [char_to_idx.get(char, 0) for char in start_text if char in char_to_idx]
    
    # Pad or truncate to match n if necessary
    if len(input_indices) < n:
        # Pad with first character if too short
        input_indices = input_indices + [input_indices[0] if input_indices else 0] * (n - len(input_indices))
    elif len(input_indices) > n:
        # Truncate if too long
        input_indices = input_indices[:n]
    
    # Convert to tensor
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate text
    with torch.no_grad():
        # Get initial output and hidden state
        output, hidden = model(input_tensor)
        
        # Initialize result with start_text
        generated_text = start_text
        
        # Generate k additional characters
        for _ in range(k):
            # Get the last prediction
            last_char_logits = output[0, -1, :].unsqueeze(0)
            
            # Sample next character
            next_char_idx = sample_from_output(last_char_logits, temperature)
            next_char = idx_to_char[next_char_idx.item()]
            
            # Add to result
            generated_text += next_char
            
            # Prepare input for next step (just the new character)
            next_input = torch.tensor([[next_char_idx.item()]], dtype=torch.long).to(device)
            
            # Get next prediction
            output, hidden = model(next_input, hidden)
    
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