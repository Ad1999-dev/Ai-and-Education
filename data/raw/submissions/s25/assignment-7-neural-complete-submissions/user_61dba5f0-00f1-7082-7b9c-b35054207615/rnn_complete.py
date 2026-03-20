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
        self.examples = []
        self.targets = []
        # Create overlapping sequences with stride
        for i in range(0, len(data) - sequence_length, stride):
            self.examples.append(data[i:i + sequence_length])
            self.targets.append(data[i + 1:i + sequence_length + 1])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = torch.tensor(self.examples[idx], dtype=torch.long)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

# ===================== Model =====================
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_dim)
        # RNN parameters
        self.W_xh = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        # Output projection
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, hidden):
        # x: [batch_size, seq_len]
        x_embed = self.embedding(x)  # [b, l, e]
        b, l, e = x_embed.size()
        x_embed = x_embed.transpose(0, 1)  # [l, b, e]
        if hidden is None:
            h_t_minus_1 = self.init_hidden(b)
        else:
            h_t_minus_1 = hidden
        outputs = []
        for t in range(l):
            x_t = x_embed[t]  # [b, e]
            # RNN recurrence: h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)
            h_t = torch.tanh(x_t @ self.W_xh + h_t_minus_1 @ self.W_hh + self.b_h)
            outputs.append(h_t)
            h_t_minus_1 = h_t
        # Stack and reshape
        hidden_seq = torch.stack(outputs)        # [l, b, h]
        hidden_seq = hidden_seq.transpose(0, 1)  # [b, l, h]
        final_hidden = h_t.clone()               # [b, h]
        # Project to vocabulary logits
        logits = hidden_seq @ self.W_hy + self.b_y  # [b, l, vocab_size]
        return logits, final_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, device=next(self.parameters()).device)

# ===================== Training Setup =====================
device = 'cpu'
print(f"Using device: {device}")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z.,!?;:()\[\] ]+', '', text)
    return text

# Simple sequence for debugging/train
#sequence = "abcdefghijklmnopqrstuvwxyz" * 100
sequence = read_file("warandpeace.txt")  # Uncomment for real data

vocab = sorted(set(sequence))
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

data = [char_to_idx[ch] for ch in sequence]

# Hyperparameters (adjust once code runs)
sequence_length = 50
stride = 8
embedding_dim = 30
hidden_size = 256
learning_rate = 0.01
num_epochs = 2
batch_size = 64
vocab_size = len(vocab)

# Initialize model, loss, optimizer
model = CharRNN(input_size=vocab_size,
                hidden_size=hidden_size,
                output_size=vocab_size,
                embedding_dim=embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Prepare data
data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
train_data = data_tensor[:train_size].tolist()
test_data = data_tensor[train_size:].tolist()

train_dataset = CharDataset(train_data, sequence_length, stride, vocab_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

test_dataset = CharDataset(test_data, sequence_length, stride, vocab_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# ===================== Training Loop =====================
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    hidden = None
    for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        # hidden = None
        optimizer.zero_grad()
        logits, hidden = model(batch_inputs, hidden)
        # detach hidden to prevent backprop through entire history
        hidden = hidden.detach()
        # compute loss over all timesteps
        loss = criterion(logits.view(-1, vocab_size), batch_targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}")

    # Evaluation on test set
    model.eval()
    test_loss = 0.0
    hidden = None
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            hidden = None
            logits, hidden = model(batch_inputs, hidden)
            hidden = hidden.detach()
            loss = criterion(logits.view(-1, vocab_size), batch_targets.view(-1))
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f"Epoch {epoch}, Test Loss:  {avg_test_loss:.4f}\n")

# ===================== Text Generation =====================
def sample_from_output(logits, temperature=1.0):
    if temperature <= 0:
        temperature = 1e-8
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_text(model, start_text, n, k, temperature=1.0):
    model.eval()
    with torch.no_grad():
        hidden = None
        # Prime the model with start_text
        for ch in start_text.lower():
            idx = char_to_idx.get(ch, None)
            if idx is None:
                continue
            inp = torch.tensor([[idx]], dtype=torch.long).to(device)
            _, hidden = model(inp, hidden)

        generated = start_text
        # Generate k new characters
        for _ in range(k):
            last_idx = char_to_idx.get(generated[-1].lower(), 0)
            inp = torch.tensor([[last_idx]], dtype=torch.long).to(device)
            logits, hidden = model(inp, hidden)
            next_logits = logits[:, -1, :]
            next_idx = sample_from_output(next_logits, temperature).item()
            generated += idx_to_char[next_idx]
    return generated

print("Training complete. Now you can generate text.")
while True:
    start_text = input("Enter the initial text (n characters, or 'exit' to quit): ")
    if start_text.lower() == 'exit':
        print("Exiting...")
        break
    k = int(input("Enter the number of characters to generate: "))
    temp_in = input("Enter the temperature value (1.0 is default, >1 is more random): ")
    temperature = float(temp_in) if temp_in else 1.0
    result = generate_text(model, start_text, len(start_text), k, temperature)
    print(f"Generated text: {result}")
