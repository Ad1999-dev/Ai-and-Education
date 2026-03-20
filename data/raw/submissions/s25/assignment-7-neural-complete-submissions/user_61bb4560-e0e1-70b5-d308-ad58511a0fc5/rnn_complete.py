
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import re

# ===================== Dataset =====================
class CharDataset(Dataset):
    def __init__(self, data, sequence_length, stride):
        self.data = data
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequences = []
        self.targets = []
        
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
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_e = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_out = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.b_out = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, hidden):
        x_embed = self.embedding(x)  # [b, l, e]
        b, l, _ = x_embed.size()
        x_embed = x_embed.transpose(0, 1)  # [l, b, e]
        if hidden is None:
            h_t_minus_1 = self.init_hidden(b)
        else:
            h_t_minus_1 = hidden

        output = []
        for t in range(l):
            x_t = x_embed[t]
            h_t = torch.tanh(torch.matmul(h_t_minus_1, self.W_h.t()) + torch.matmul(x_t, self.W_e) + self.b_h)
            output.append(h_t)
            h_t_minus_1 = h_t

        logits = torch.matmul(torch.stack(output), self.W_out) + self.b_out
        return logits, h_t_minus_1

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

# ===================== Main Script =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and process text
def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z.,!?;:()\[\] ]+', '', text)
    return text

# Debug/test sequence
sequence = "abcdefghijklmnopqrstuvwxyz" * 100
# sequence = read_file("warandpeace.txt")  # Uncomment to use real data

vocab = sorted(set(sequence))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
data = [char_to_idx[char] for char in sequence]

# Hyperparameters
sequence_length = 100
stride = 10
embedding_dim = 30
hidden_size = 128
learning_rate = 0.001
num_epochs = 10
batch_size = 64

# Prepare data
data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(0.9 * len(data_tensor))
train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]

train_dataset = CharDataset(train_data, sequence_length, stride)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Model setup
vocab_size = len(vocab)
model = CharRNN(vocab_size, hidden_size, vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    hidden = None
    for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        output, hidden = model(batch_inputs, hidden)
        hidden = hidden.detach()

        loss = criterion(output.view(-1, output.size(2)), batch_targets.view(-1))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# ===================== Evaluation =====================
def test_model(model, test_data, sequence_length):
    model.eval()
    hidden = None
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(test_data) - sequence_length - 1, sequence_length):
            x = test_data[i:i+sequence_length].unsqueeze(0).to(device)
            y = test_data[i+1:i+sequence_length+1].unsqueeze(0).to(device)
            output, hidden = model(x, hidden)
            loss = criterion(output.view(-1, output.size(2)), y.view(-1))
            total_loss += loss.item()
    return total_loss / (len(test_data) // sequence_length)

test_loss = test_model(model, test_data, sequence_length)
print(f"Test Loss: {test_loss:.4f}")

# ===================== Text Generation =====================
def sample_from_output(logits, temperature=1.0):
    scaled_logits = logits / max(temperature, 1e-8)
    probabilities = F.softmax(scaled_logits, dim=-1)
    sampled_idx = torch.multinomial(probabilities, 1)  # shape: [1, 1]
    return sampled_idx[0][0]  # scalar index

def generate_text(model, start_text, k, temperature=1.0):
    model.eval()
    input_seq = torch.tensor([char_to_idx[c] for c in start_text], dtype=torch.long).unsqueeze(0).to(device)
    generated_text = start_text
    hidden = None
    with torch.no_grad():
        for _ in range(k):
            output, hidden = model(input_seq, hidden)
            next_char_logits = output[:, -1, :]
            next_char_idx = sample_from_output(next_char_logits, temperature)
            next_char = idx_to_char[next_char_idx.item()]
            generated_text += next_char
            input_seq = torch.cat((input_seq, next_char_idx.view(1, 1)), dim=1)
    return generated_text

print("Training complete. You can now generate text!")
while True:
    start_text = input("Enter the initial text (n characters, or 'exit' to quit): ")
    
    if start_text.lower() == 'exit':
        print("Exiting...")
        break
    
    n = len(start_text) 
    k = int(input("Enter the number of characters to generate: "))
    temperature_input = input("Enter the temperature value (1.0 is default, >1 is more random): ")
    temperature = float(temperature_input) if temperature_input else 1.0
    
    completed_text = generate_text(model, start_text, k, temperature)
    
    print(f"Generated text: {completed_text}")
