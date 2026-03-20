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
        sequence = self.sequences[idx].clone().detach()
        target = self.targets[idx].clone().detach()
        return sequence, target

# ===================== Model =====================
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.W_ih = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
        

    def forward(self, x, hidden):
        """
        x in [b, l] # b is batch_size and l is sequence length
        """
        x_embed = self.embedding(x)  # [b, l, e]
        b, l, _ = x_embed.size()
        x_embed = x_embed.transpose(0, 1)  # [l, b, e]

        if hidden is None:
            h_t_minus_1 = self.init_hidden(b)  # [b, h]
        else:
            h_t_minus_1 = hidden  # [b, h]

        output = []

        for t in range(l):
            x_t = x_embed[t]  # [b, e]
            h_t = torch.tanh(
                x_t @ self.W_ih + h_t_minus_1 @ self.W_hh + self.b_h  # [b, h]
            )
            output.append(h_t)
            h_t_minus_1 = h_t

        output = torch.stack(output)  # [l, b, h]
        output = output.transpose(0, 1)  # [b, l, h]

        logits = self.output_layer(output)  # [b, l, vocab_size]
        final_hidden = h_t_minus_1.detach()  # [b, h]

        return logits, final_hidden
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(next(self.parameters()).device)

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
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
data = [char_to_idx[char] for char in sequence]


sequence_length = 100
stride = 10
embedding_dim = 64
hidden_size = 256
learning_rate = 0.003
num_epochs = 3
batch_size = 64
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)

train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]

train_dataset = CharDataset(train_data, sequence_length, stride, output_size)
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
        loss = criterion(output.view(-1, output_size), batch_targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
# ===================== Test Loop =====================
test_dataset = CharDataset(test_data, sequence_length, stride, output_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

model.eval()
test_loss = 0
hidden = None

with torch.no_grad():
    for batch_inputs, batch_targets in test_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        output, hidden = model(batch_inputs, hidden)
        hidden = hidden.detach()
        loss = criterion(output.view(-1, output_size), batch_targets.view(-1))
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_loader):.4f}")
model.train()

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
    Generate text from a trained character-level RNN.

    model: trained CharRNN model
    start_text: string of length n used as seed input
    n: number of seed characters
    k: number of characters to generate
    temperature: randomness control for sampling
    """
    model.eval()
    input_indices = [char_to_idx[ch] for ch in start_text.lower()[-n:]]  # last n chars
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)  # [1, n]
    
    hidden = None
    generated = start_text

    with torch.no_grad():
        # Feed the seed text first
        for i in range(n):
            _, hidden = model(input_tensor[:, i:i+1], hidden)

        current_input = input_tensor[:, -1:]  # last char as input

        for _ in range(k):
            output, hidden = model(current_input, hidden)  # output: [1, 1, vocab_size]
            logits = output[:, -1, :]  # [1, vocab_size]
            sampled_idx = sample_from_output(logits, temperature=temperature)  # [1, 1]
            next_char = idx_to_char[sampled_idx.item()]
            generated += next_char

            current_input = sampled_idx  # feed back as input

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
    