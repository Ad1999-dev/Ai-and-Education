
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        # Keep only lowercase letters and standard punctuation (.,!?;:()[])
        text = re.sub(r'[^a-z.,!?;:()\[\] ]+', '', text)
    return text

sequence = read_file("warandpeace.txt")

#TODO: Create a list of unique characters from the text sequence
vocab = sorted(set(sequence))

#TODO: Create two dictionaries for character-index mappings that map each character in vocab to a unique index and vice versa
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}

#TODO: Convert the entire text based data into numerical data
data = [char_to_idx[char] for char in sequence]

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

#TODO: Set your model's hyperparameters

sequence_length = 100  # Length of each input sequence
stride = 1             # Stride for creating sequences
embedding_dim = 64     # Dimension of character embeddings
hidden_size = 128      # Number of features in the hidden state of the RNN
learning_rate = 0.001  # Learning rate for the optimizer
num_epochs = 10        # Number of epochs to train
batch_size = 64        # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

data_tensor = torch.tensor(data, dtype=torch.long)

#TODO: Convert the data into a pytorch tensor and split the data into 90:10 ratio
train_size = int(len(data_tensor) * 0.9)
train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]

train_dataset = CharDataset(train_data, sequence_length, stride, vocab_size)
test_dataset = CharDataset(test_data, sequence_length, stride, vocab_size)

#TODO: Initialize the training and testing data loader with batching and shuffling equal to True for training (and shuffling = False for testing)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)    

total_batches = len(train_loader)

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=30):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(output_size, embedding_dim)
        self.W_e = nn.Parameter(torch.randn(hidden_size, embedding_dim) * 0.01)  # Smaller std
        self.b_e = nn.Parameter(torch.zeros(hidden_size))
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)  # Smaller std
        self.b_h = nn.Parameter(torch.zeros(hidden_size)) 
        #TODO: set the fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

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
            # RNN equation from the lecture 
            # We add a bias as well to expand the range of learnable functions
            h_t = torch.tanh(x_embed[t] @ self.W_e.T + self.b_e + h_t_minus_1 @ self.W_h.T + self.b_h) # [b, e]
            output.append(h_t)
            h_t_minus_1 = h_t
        output = torch.stack(output) # [l, b, e]
        output = output.transpose(0, 1) # [b, l, e]
        final_hidden = h_t.clone() # [b, h]
        logits = self.fc(output) # [b, l, vocab_size=v] 
        return logits, final_hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

#TODO: Initialize your RNN model
model = CharRNN(input_size, hidden_size, output_size, embedding_dim)

#TODO: Define the loss function (use cross entropy loss)
criterion = nn.CrossEntropyLoss()

#TODO: Initialize your optimizer passing your model parameters and training hyperparameters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss, total_correct, total_predictions = 0, 0, 0

    hidden = model.init_hidden(batch_size)

    for batch_idx, (batch_inputs, batch_targets) in tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        output, hidden = model(batch_inputs, hidden)

        hidden = hidden.detach()

        loss = criterion(output.view(-1, output_size), batch_targets.view(-1))  # Flatten the outputs and targets for CrossEntropyLoss
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            # Calculate accuracy
            _, predicted_indices = torch.max(output, dim=2)  # Predicted characters

            total_correct += (predicted_indices == batch_targets).sum().item()
            total_predictions += batch_targets.size(0) * batch_targets.size(1)  # Total items in this batch

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_predictions * 100  # Convert to percentage
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

