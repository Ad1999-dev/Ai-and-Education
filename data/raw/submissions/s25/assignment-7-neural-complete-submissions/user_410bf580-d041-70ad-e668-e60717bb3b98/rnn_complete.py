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
        # Initialize parameters
        self.hidden_size = hidden_size    
        self.input_size = input_size
        self.output_size = output_size

        # Weight matrices
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.W_xh = nn.Parameter(torch.randn(hidden_size, embedding_dim) * 0.01)  # Input to hidden
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)  # Hidden to hidden
        self.b_h = nn.Parameter(torch.zeros(hidden_size))  # Hidden bias
        self.W_hy = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)  # Hidden to output
        self.b_y = nn.Parameter(torch.zeros(output_size))  # Output bias

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
            x_t = x_embed[t]  # Get the embedding for timestep t
            h_t = torch.tanh(
                x_t @ self.W_xh.T + h_t_minus_1 @ self.W_hh.T + self.b_h
            )
            output.append(h_t)  # Capture current hidden state
            h_t_minus_1 = h_t  # Update hidden state for next timestep
        # Stack outputs into a tensor; shape will be [l, b, h]
        output = torch.stack(output)

        # Transpose output to [b, l, h] if needed, for batch processing later
        output = output.transpose(0, 1)

        # Save the final hidden state
        final_hidden = h_t_minus_1.clone()  # Ensure to call clone as a method

        # Calculate logits (output predictions).
        # Assuming self.W_hy is the hidden to output weight matrix and self.b_y is the bias for output
        logits = output @ self.W_hy.T + self.b_y  # Shape [b, l, vocab_size]
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
#sequence = "abcdefghijklmnopqrstuvwxyz" * 100
sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))

char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}

data = [char_to_idx[char] for char in sequence]

# Understand and adjust the hyperparameters once the code is running
# For "abcdefghijklmnopqrstuvwxyz"

"""sequence_length = 30   # Length of each input sequence
stride = 2             # Stride for creating sequences
embedding_dim = 32      # Dimension of character embeddings
hidden_size = 128        # Number of features in the hidden state of the RNN
learning_rate = 0.01     # Learning rate for the optimizer
num_epochs = 10         # Number of epochs to train
batch_size = 64        # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)
"""


# For war and peace
# 
sequence_length = 100   # Length of each input sequence
stride = 4            # Stride for creating sequences
embedding_dim = 128      # Dimension of character embeddings
hidden_size = 256        # Number of features in the hidden state of the RNN
learning_rate = 0.01     # Learning rate for the optimizer
num_epochs = 1         # Number of epochs to train
batch_size = 128       # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)



model = CharRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
#Split the data into 90:10 ratio with PyTorch indexing
train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]

train_dataset = CharDataset(train_data, sequence_length, stride, output_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    hidden = model.init_hidden(batch_size)  # Re-initialize hidden state for each epoch
    hidden = hidden.to(device)  # Ensure hidden state is on the correct device

    for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        output, hidden = model(batch_inputs, None)
        # Detach removes the hidden state from the computational graph.
        # This is prevents backpropagating through the full history, and
        # is important for stability in training RNNs
        hidden = hidden.detach()

        optimizer.zero_grad()

        # Reshape output and targets
        output = output.reshape(-1, output_size)
        batch_targets = batch_targets.reshape(-1)

        loss = criterion(output, batch_targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
# Implement a test loop to evaluate the model on the test set

test_dataset = CharDataset(test_data, sequence_length, stride, output_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

model.eval()  # Set the model to evaluation mode

total_test_loss = 0
correct_predictions = 0
num_samples = 0

# Iterate over the test data
with torch.no_grad():  # No gradients needed for testing
    hidden = None  # Reset hidden for evaluation
    for batch_inputs, batch_targets in tqdm(test_loader, desc="Testing"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        # Forward pass
        output, hidden = model(batch_inputs, hidden)

        # Reshape output and targets for loss calculation
        output = output.reshape(-1, output_size)
        batch_targets = batch_targets.reshape(-1)

        # Calculate loss
        loss = criterion(output, batch_targets)
        total_test_loss += loss.item()

        # Calculate predictions
        predicted_indices = torch.argmax(output, dim=1)
        correct_predictions += (predicted_indices == batch_targets).sum().item()  # Count correct predictions
        num_samples += batch_targets.size(0)

# Average loss 
average_test_loss = total_test_loss / len(test_loader)
accuracy = correct_predictions / num_samples * 100  # Convert to percentage

print(f"Test Loss: {average_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")



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
    model.eval()
    start_text = start_text.lower()
    
    # Encode the start text into indices
    input_sequence = torch.tensor([char_to_idx[char] for char in start_text], dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    generated_text = ""

    for _ in range(k):
        output, hidden = model(input_sequence, hidden)

        # We only want the last time step's output
        logits = output[:, -1, :]  # Shape: [batch_size=1, vocab_size]

        # Sample from the output distribution
        next_idx = sample_from_output(logits, temperature).item()

        # Map index back to character
        next_char = idx_to_char[next_idx]

        # Append to generated text
        generated_text += next_char

        # Prepare input for next iteration (slide window)
        new_input = torch.tensor([[next_idx]], dtype=torch.long).to(device)
        input_sequence = torch.cat((input_sequence[:, 1:], new_input), dim=1)

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