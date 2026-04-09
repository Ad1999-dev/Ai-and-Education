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
        self.W_ih = nn.Linear(embedding_dim, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, output_size)     

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
            #  TODO: Implement forward pass for a single RNN timestamp, append the hidden to the output
            x_t = x_embed[t]  
            h_t = torch.tanh(self.W_ih(x_t) + self.W_hh(h_t_minus_1))  
            output.append(h_t)
            h_t_minus_1 = h_t  
        output = torch.stack(output)
        output = output.transpose(0, 1) 
        
        # TODO set these values after completing the loop above
        final_hidden = h_t 
        logits = self.W_ho(output) 
        return logits, final_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

# ===================== Training =====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z.,!?;:()\[\] ]+', '', text)
    return text

# To debug your model you should start with a simple sequence an RNN should predict this perfectly
sequence = "abcdefghijklmnopqrstuvwxyz" * 100
# sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
data = [char_to_idx[char] for char in sequence]

# TODO Understand and adjust the hyperparameters once the code is running
sequence_length = 50 
stride = 5            
embedding_dim = 8
hidden_size = 64
learning_rate = 0.01    
num_epochs = 3
batch_size = 32        
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
#TODO: Split the data into 90:10 ratio with PyTorch indexing

vocab = sorted(set(sequence))
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
data = [char_to_idx[char] for char in sequence]

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
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

        logits, hidden = model(batch_inputs, hidden)
        hidden = hidden.detach()

        # compute loss & step
        loss = criterion(logits.view(-1, vocab_size), batch_targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
# TODO: Implement a test loop to evaluate the model on the test set
test_dataset = CharDataset(test_data, sequence_length, stride, output_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs, None)
        test_loss += criterion(logits.view(-1, vocab_size), targets.view(-1)).item()
if len(test_loader) > 0:
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

# ===================== Text Generation =====================
def sample_from_output(logits, temperature=1.0):
    """
    Sample from the logits with temperature scaling.
    logits: Tensor of shape [batch_size, vocab_size] (raw scores, before softmax)
    temperature: a float controlling the randomness (higher = more random)
    """
    if temperature <= 0:
        temperature = 0.00000001
    scaled_logits = logits / temperature 
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(scaled_logits, dim=1)
    
    # Sample from the probability distribution
    sampled_idx = torch.multinomial(probabilities, 1)
    return sampled_idx

def generate_text(model, start_text, n, k, temperature=1.0):
    start_text = start_text.lower()
    # Filter out characters not in the trained vocabulary.
    filtered_text = ''.join(ch for ch in start_text if ch in char_to_idx)
    if len(filtered_text)==0:
        print("No valid characters found in input!")
        return start_text
    idxs = [char_to_idx[ch] for ch in filtered_text]
    inp = torch.tensor(idxs, dtype=torch.long).unsqueeze(0).to(device)  # [1, n]
    hidden = None
    _, hidden = model(inp, hidden)

    generated = list(filtered_text)
    # prime the next input with the last char
    inp = inp[:, -1].unsqueeze(1)  # [1, 1]
    for _ in range(k):
        logits, hidden = model(inp, hidden)     # logits [1,1,v]
        next_logits = logits[:, -1, :]           # [1, v]
        idx = sample_from_output(next_logits, temperature).item()
        generated.append(idx_to_char[idx])
        inp = torch.tensor([[idx]], dtype=torch.long).to(device)

    return "".join(generated)

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