import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
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
        # Input projection: input-to-hidden weights
        self.W_xh = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        # Output projection: hidden-to-output logits
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(output_size))


    def forward(self, x, hidden):
        """
        x in [b, l] # b is batch_size and l is sequence length
        """
        x_embed = self.embedding(x)
        b, l, _ = x_embed.size()
        x_embed = x_embed.transpose(0, 1)       
        if hidden is None:
            h_t_minus_1 = self.init_hidden(b)
        else:
            h_t_minus_1 = hidden

        output = []
        for t in range(l):
            h_t = torch.tanh(x_embed[t] @ self.W_xh + h_t_minus_1 @ self.W_hh + self.b_h)
            output.append(h_t)
            h_t_minus_1 = h_t

        output = torch.stack(output)
        output = output.transpose(0, 1)

        final_hidden = h_t_minus_1.clone()
        logits = output @ self.W_hy + self.b_y
        return logits, final_hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

# ===================== Training =====================
device = 'mps' if torch.mps.is_available() else 'cpu'
print(f"Using device: {device}")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z.,!?;:()\[\] ]+', '', text)
    return text

# To debug your model you should start with a simple sequence an RNN should predict this perfectly
# sequence = "abcdefghijklmnopqrstuvwxyz" * 100
sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}
data = [char_to_idx[char] for char in sequence]


sequence_length = 50 # Length of each input sequence
stride = 15       # Stride for creating sequences
embedding_dim = 32      # Dimension of character embeddings
hidden_size = 128        # Number of features in the hidden state of the RNN
learning_rate = 0.01    # Learning rate for the optimizer
num_epochs = 3       # Number of epochs to train
batch_size = 64        # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

 # Prepare datasets by splitting the full CharDataset
data_tensor = torch.tensor(data, dtype=torch.long)
train_len = int(len(data_tensor) * 0.9)
train_data = data_tensor[:train_len]
test_data = data_tensor[train_len:]

train_dataset = CharDataset(train_data, sequence_length, stride, output_size)
test_dataset = CharDataset(test_data, sequence_length, stride, output_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    hidden = None
    for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        # Detach removes the hidden state from the computational graph.
        # This is prevents backpropagating through the full history, and
        # is important for stability in training RNNs
        optimizer.zero_grad()
        output, hidden = model(batch_inputs, hidden)
        hidden = hidden.detach()
        loss = criterion(output.view(-1, output_size), batch_targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss_avg = total_loss / len(train_loader)


    print(f"Epoch {epoch+1}, Loss: {train_loss_avg:.4f}")

# Test the model
model.eval()
total_test_loss = 0
hidden = None
with torch.no_grad():
    for batch_inputs, batch_targets in test_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        output, hidden = model(batch_inputs, hidden)
        hidden = hidden.detach()
        loss = criterion(output.view(-1, output_size), batch_targets.view(-1))
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}")

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
    model.eval()
    # Convert start_text to indices
    input_idxs = [char_to_idx[ch] for ch in start_text]
    input_idxs = torch.tensor(input_idxs, dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    generated = start_text

    for _ in range(k):
        logits, hidden = model(input_idxs, hidden)
        next_logits = logits[:, -1, :]
        next_idx = sample_from_output(next_logits, temperature)

        # Fix: next_idx should be (1,) not (1,1) shape
        next_idx = next_idx.squeeze(1)

        next_char = idx_to_char[next_idx.item()]
        generated += next_char

        input_idxs = torch.cat([input_idxs[:, 1:], next_idx.unsqueeze(0)], dim=1)

    return generated

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