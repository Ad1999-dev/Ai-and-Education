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
            self.sequences.append(data[i : i + sequence_length])
            self.targets.append(data[i + 1 : i + sequence_length + 1])

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
        self.weight_input_hidden = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.weight_hidden_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias_hidden = nn.Parameter(torch.zeros(hidden_size))
        self.weight_hidden_output = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        x in [b, l] # b is batch_size and l is sequence length
        """
        x_embed = self.embedding(
            x
        )  # [b=batch_size, l=sequence_length, e=embedding_dim]
        b, l, _ = x_embed.size()
        x_embed = x_embed.transpose(0, 1)  # [l, b, e]
        if hidden is None:
            h_t_minus_1 = self.init_hidden(b)
        else:
            h_t_minus_1 = hidden
        output = []
        for t in range(l):
            #  TODO: Implement forward pass for a single RNN timestamp, append the hidden to the output
            x_t = x_embed[t]  # Get embedding at current timestamp [b, e]
            h_t = torch.tanh(
                self.weight_input_hidden(x_t)
                + self.weight_hidden_hidden(h_t_minus_1)
                + self.bias_hidden
            )
            h_t_minus_1 = h_t
            output.append(h_t)
        output = torch.stack(output)  # [l, b, e]
        output = output.transpose(0, 1)  # [b, l, e]

        # Create a deep copy of the final hidden state
        final_hidden = h_t_minus_1.clone()  # [b, h]
        logits = self.weight_hidden_output(output)  # [b, l, vocab_size=v]
        return logits, final_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)


# ===================== Training =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r"[^a-z.,!?;:()\[\] ]+", "", text)
    return text



# To debug your model you should start with a simple sequence an RNN should predict this perfectly
sequence = "abcdefghijklmnopqrstuvwxyz" * 10000
# sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))
char_to_idx = {char: i for i, char in enumerate(vocab)} 
idx_to_char = {i: char for i, char in enumerate(vocab)}  
data = [char_to_idx[char] for char in sequence]

# TODO Understand and adjust the hyperparameters once the code is running
sequence_length = 1000  # Length of each input sequence
stride = 50  # Stride for creating sequences
embedding_dim = 100 # Dimension of character embeddings
hidden_size = 256  # Number of features in the hidden state of the RNN
learning_rate = 0.01  # Learning rate for the optimizer
num_epochs = 1  # Number of epochs to train
batch_size = 64  # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
# TODO: Split the data into 90:10 ratio with PyTorch indexing

train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]

train_dataset = CharDataset(train_data, sequence_length, stride, output_size)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    hidden = None
    train_correct = 0
    train_total = 0
    for batch_inputs, batch_targets in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"
    ):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        output, hidden = model(batch_inputs, hidden)
        # Detach removes the hidden state from the computational graph.
        # This is prevents backpropagating through the full history, and
        # is important for stability in training RNNs
        hidden = hidden.detach()

        b, l, v = output.size()
        output_flat = output.reshape(-1, v) 
        targets_flat = batch_targets.reshape(-1)
        loss = criterion(output_flat, targets_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(output_flat, 1)
        train_total += targets_flat.size(0)
        train_correct += (predicted == targets_flat).sum().item()

    train_accuracy = train_correct / train_total * 100
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

# Test the model
# TODO: Implement a test loop to evaluate the model on the test set
test_dataset = CharDataset(test_data, sequence_length, stride, output_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

model.eval()  # Set the model to evaluation mode
test_loss = 0
hidden = None
test_correct = 0
test_total = 0

with torch.no_grad():  # No need to track gradients during evaluation
    for batch_inputs, batch_targets in tqdm(test_loader, desc="Testing"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        
        output, hidden = model(batch_inputs, hidden)
        hidden = hidden.detach()
        
        b, l, v = output.size()
        output_flat = output.reshape(-1, v)
        targets_flat = batch_targets.reshape(-1)
        loss = criterion(output_flat, targets_flat)
        test_loss += loss.item()
        
        # Calculate test accuracy
        _, predicted = torch.max(output_flat, 1)
        test_total += targets_flat.size(0)
        test_correct += (predicted == targets_flat).sum().item()

if len(test_loader) > 0:
    test_accuracy = test_correct / test_total * 100
    print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")

else:
    print("Test Loss: N/A (no test batches)")
model.train()  # Set the model back to training mode

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
    char_indices = [char_to_idx.get(c, 0) for c in start_text]  # Use 0 as default for unknown chars
    generated_text = ""
    hidden = None
    model.eval()
    
    with torch.no_grad():
        for _ in range(k):
            x = torch.tensor([char_indices], dtype=torch.long).to(device)
            output, hidden = model(x, hidden)
            last_char_logits = output[0, -1, :]
            
            # Sample next character index using temperature
            next_char_idx = sample_from_output(last_char_logits.unsqueeze(0), temperature)
            next_char_idx = next_char_idx.item()
            
            # Convert index back to character and add to generated text
            next_char = idx_to_char[next_char_idx]
            generated_text += next_char
            
            # Update input sequence by sliding window: remove first char, add new char
            char_indices = char_indices[1:] + [next_char_idx]
    model.train()
    
    return generated_text


print("Training complete. Now you can generate text.")
while True:
    start_text = input("Enter the initial text (n characters, or 'exit' to quit): ")
    
    if start_text.lower() == 'exit':
        print("Exiting...")
        break

    n = len(start_text)
    k = int(input("Enter the number of characters to generate: "))
    temperature_input = input(
        "Enter the temperature value (1.0 is default, >1 is more random): "
    )
    temperature = float(temperature_input) if temperature_input else 1.0

    completed_text = generate_text(model, start_text, n, k, temperature)

    print(f"Generated text: {completed_text}")