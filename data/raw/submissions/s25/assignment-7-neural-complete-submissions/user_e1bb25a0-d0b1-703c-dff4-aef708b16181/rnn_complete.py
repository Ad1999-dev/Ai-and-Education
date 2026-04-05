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

        # 1.  Embedding layer (already provided)
        self.embedding = nn.Embedding(output_size, embedding_dim)

        # 2.  Recurrent weights   h_t = tanh( x_t W_xh  +  h_{t-1} W_hh  +  b_h )
        self.W_xh = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size,     hidden_size) * 0.01)
        self.b_h  = nn.Parameter(torch.zeros(hidden_size))

        # 3.  Projection to vocabulary logits   y_t = h_t W_hy + b_y
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.b_y  = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, hidden=None):
        x_embed = self.embedding(x)               # [b, l, e]
        b, l, _ = x_embed.shape
        x_embed = x_embed.transpose(0, 1)         # [l, b, e]

        if hidden is None:
            h_t_minus_1 = self.init_hidden(b)
        else:
            h_t_minus_1 = hidden                  # [b, h]

        outputs = []
        for t in range(l):
            x_t = x_embed[t]                      # [b, e]
            h_t = torch.tanh(
                    x_t @ self.W_xh +             # [b, h]
                    h_t_minus_1 @ self.W_hh +
                    self.b_h)
            outputs.append(h_t)
            h_t_minus_1 = h_t                     # prepare for next step

        outputs = torch.stack(outputs)            # [l, b, h]
        outputs = outputs.transpose(0, 1)         # [b, l, h]

        final_hidden = h_t_minus_1.clone()        # [b, h]
        logits = outputs @ self.W_hy + self.b_y   # [b, l, v]
        return logits, final_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


# ===================== Training =====================
device = 'cpu'
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
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
data = [char_to_idx[char] for char in sequence]

# TODO Understand and adjust the hyperparameters once the code is running
sequence_length = 100 # Length of each input sequence
stride = 3            # Stride for creating sequences
embedding_dim = 32      # Dimension of character embeddings
hidden_size = 128        # Number of features in the hidden state of the RNN
learning_rate = 0.002    # Learning rate for the optimizer
num_epochs = 1         # Number of epochs to train
batch_size = 256        # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
#TODO: Split the data into 90:10 ratio with PyTorch indexing
train_data = data_tensor[:train_size]
test_data  = data_tensor[train_size:]

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

        logits = output.reshape(-1, vocab_size)           # [b*l, v]
        targets = batch_targets.reshape(-1)               # [b*l]
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
# ---------------- Test loop ----------------
with torch.no_grad():
    test_dataset = CharDataset(test_data, sequence_length, stride, output_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loss = 0
    for batch_inputs, batch_targets in test_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        logits, _ = model(batch_inputs, None)
        loss = criterion(logits.reshape(-1, vocab_size), batch_targets.reshape(-1))
        test_loss += loss.item()
    print(f"Test‑set loss: {test_loss/len(test_loader):.4f}")


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
    model.eval()

    # convert start text to indices
    indices = [char_to_idx.get(c, 0) for c in start_text]
    hidden = None

    for _ in range(k):
        # feed the last n characters
        input_seq = torch.tensor(indices[-n:], dtype=torch.long).unsqueeze(0)  # [1, n]
        logits, hidden = model(input_seq, hidden)
        last_logits = logits[0, -1]  # [v]
        next_idx = sample_from_output(last_logits.unsqueeze(0), temperature)
        indices.append(next_idx.item())

    return ''.join(idx_to_char[i] for i in indices)

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