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
        self.rnn = nn.RNN(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)  # to project hidden state to vocab size



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
            h_t, h_t_minus_1 = self.rnn(x_embed[t].unsqueeze(0), h_t_minus_1)
            output.append(h_t)
            #  TODO: Implement forward pass for a single RNN timestamp, append the hidden to the output 
    
        output = torch.stack(output) # [l, b, e]
        output = output.transpose(0, 1) # [b, l, e]
        
        # TODO set these values after completing the loop above
        # final_hidden = None # [b, h] 
        # logits = None # [b, l, vocab_size=v] 
        logits = self.fc(output)  # [b, l, vocab_size]
        final_hidden = h_t
        return logits, final_hidden
    
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)  # [1, batch_size, hidden_size]

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
char_to_idx = {char: idx for idx, char in enumerate(vocab)} # TODO: Create a mapping from characters to indices
idx_to_char = {idx: char for idx, char in enumerate(vocab)} # TODO: Create the reverse mapping
data = [char_to_idx[char] for char in sequence]

# TODO Understand and adjust the hyperparameters once the code is running
sequence_length = 10 # Length of each input sequence
stride = 1            # Stride for creating sequences
embedding_dim = 16      # Dimension of character embeddings
hidden_size = 64        # Number of features in the hidden state of the RNN
learning_rate = 0.001    # Learning rate for the optimizer
num_epochs = 1         # Number of epochs to train
batch_size = 64        # Batch size for training
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
test_data = data_tensor[train_size:]

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

        output, hidden = model(batch_inputs, hidden)
        # Detach removes the hidden state from the computational graph.
        # This is prevents backpropagating through the full history, and
        # is important for stability in training RNNs
        hidden = hidden.detach()

        # TODO compute the loss, backpropagate gradients, and update total_loss
        # Compute the loss
        loss = criterion(output.view(-1, vocab_size), batch_targets.view(-1))
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Train loader size: {len(train_loader)}")
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
# TODO: Implement a test loop to evaluate the model on the test set
# Test the model
def evaluate(model, test_loader):
    model.eval()  # Set to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No gradients needed
        hidden = None
        for batch_inputs, batch_targets in test_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            output, hidden = model(batch_inputs, hidden)
            hidden = hidden.detach()

            loss = criterion(output.view(-1, vocab_size), batch_targets.view(-1))
            total_loss += loss.item()

    print(f"Test Loss: {total_loss / len(test_loader):.4f}")

# Evaluate the model on the test set
evaluate(model, test_loader)
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
    probabilities = F.softmax(scaled_logits, dim=1)  # Now probabilities should be [batch_size, vocab_size]
    
    # Extract probabilities for the last time step (last character in sequence)
    probabilities = probabilities[:, -1, :]  # Now shape is [batch_size, vocab_size]
    
    # Ensure the probabilities tensor has only 2 dimensions
    assert probabilities.dim() == 2, f"Expected 2D tensor, got {probabilities.dim()}"


    # Sample from the probability distribution
    sampled_idx = torch.multinomial(probabilities, 1)  # Take 1 sample from the distribution

    return sampled_idx.squeeze()


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
    start_idx = [char_to_idx[char] for char in start_text]
    input_tensor = torch.tensor(start_idx, dtype=torch.long).unsqueeze(0).to(device)  # Shape: [1, n]
    hidden = None
    generated_text = start_text

    for _ in range(k):
        output, hidden = model(input_tensor, hidden)
        logits = output[:, -1, :]  # Get the last character's logits

        # Sampling from the output logits
        sampled_idx = sample_from_output(logits, temperature)  # Shape: [batch_size]

        # Convert sampled_idx to the correct shape for concatenation
        sampled_idx = sampled_idx.view(1,1)  # Shape: [1, batch_size]

        # Concatenate the sampled character to the input sequence
        input_tensor = torch.cat([input_tensor, sampled_idx], dim=1)  # Concatenate along the sequence length axis

        # Convert the sampled index back to a character
        sampled_char = idx_to_char[sampled_idx.item()]
        
        generated_text += sampled_char

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