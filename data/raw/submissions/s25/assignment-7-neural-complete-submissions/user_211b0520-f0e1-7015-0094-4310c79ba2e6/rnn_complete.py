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
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)  # RNN layer
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer for output

    def forward(self, x, hidden):
        """
        x in [b, l] # b is batch_size and l is sequence length
        """
        x_embed = self.embedding(x)  # [b=batch_size, l=sequence_length, e=embedding_dim]
        output, hidden = self.rnn(x_embed, hidden)  # RNN forward pass
        logits = self.fc(output)  # [b, l, vocab_size]
        return logits, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)  # Shape for RNN

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
char_to_idx = {char: idx for idx, char in enumerate(vocab)}  # Create a mapping from characters to indices
idx_to_char = {idx: char for idx, char in enumerate(vocab)}  # Create the reverse mapping
data = [char_to_idx[char] for char in sequence]

# Adjusting hyperparameters
sequence_length = 200  # Length of each input sequence
stride = 75            # Stride for creating sequences
embedding_dim = 128      # Dimension of character embeddings
hidden_size = 128        # Number of features in the hidden state of the RNN
learning_rate = 0.003    # Learning rate for the optimizer
num_epochs = 5         # Number of epochs to train
batch_size = 64     # Batch size for training
vocab_size = len(vocab)

model = CharRNN(vocab_size, hidden_size, vocab_size, embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)

# Split the data into 90:10 ratio
train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:] 

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
        hidden = hidden.detach()

        # Compute the loss, backpropagate gradients, and update total_loss
        loss = criterion(output.view(-1, vocab_size), batch_targets.view(-1))  # Flatten for cross entropy
        optimizer.zero_grad()  # Zero gradients before backward propagation
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights
        total_loss += loss.item()  # Accumulate loss

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
def test_model(model, test_data):
    # Ensure test data is long enough for at least one sequence
    if len(test_data) < sequence_length:
        print("Test data is too short to form any sequences.")
        return
    
    test_dataset = CharDataset(test_data, sequence_length, stride, vocab_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # Check for empty test_loader
    if len(test_loader) == 0:
        print("No batches available in test_loader.")
        return

    model.eval()  # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No gradient calculation
        for batch_inputs, batch_targets in test_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            output, _ = model(batch_inputs, None)
            loss = criterion(output.view(-1, vocab_size), batch_targets.view(-1))
            total_loss += loss.item()
 # Add check to avoid division by zero
    if len(test_loader) > 0:
        print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    else:
        print("Error: No valid batches to calculate test loss.")

test_model(model, test_data)

# ===================== Text Generation =====================
def sample_from_output(logits, temperature=1.0):
    if temperature <= 0:
        temperature = 0.00000001
    scaled_logits = logits / temperature 
    probabilities = F.softmax(scaled_logits, dim=0)
    sampled_idx = torch.multinomial(probabilities, 1)
    return sampled_idx

def generate_text(model, start_text, n, k, temperature=1.0):
    model.eval()
    with torch.no_grad():
        input_data = torch.tensor([char_to_idx[char] for char in start_text]).unsqueeze(0).to(device)  # Prepare input
        hidden = model.init_hidden(1)

        generated_text = list(start_text)

        for _ in range(k):
            output, hidden = model(input_data, hidden)
            last_char_logits = output[0, -1]  # Get the output of the last character in the sequence
            predicted_idx = sample_from_output(last_char_logits, temperature)
            generated_text.append(idx_to_char[predicted_idx.item()])  # Append the predicted character to the generated text

            # Use the last character as the next input
            input_data = torch.cat((input_data[:, 1:], predicted_idx.unsqueeze(0)), dim=1)

    return ''.join(generated_text)

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