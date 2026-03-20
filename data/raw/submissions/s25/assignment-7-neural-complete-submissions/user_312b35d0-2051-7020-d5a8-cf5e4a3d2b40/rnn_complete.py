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
        # self.input_size = input_size
        
        # Initialize your model parameters as needed e.g. W_e, W_h, etc.
        # W_e has to be embedding_dim x hidden_size matrix so W_e*e_t produces a hidden_sizex1 matrix
        self.W_e = nn.Parameter(torch.randn(embedding_dim, hidden_size))
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size)) # create a hidden_size x hidden_size matrix so W_hh_{t-1} produces a 1xhidden_size matrix
        # now when we add W_e*e_t + W_hh_{t-1} it results in a 1xhidden_size vector
        # self.W_0 = nn.Parameter(torch.randn(hidden_size, output_size))
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        x in [b, l] # b is batch_size and l is sequence length
        """
        x_embed = self.embedding(x)  # [b=batch_size, l=sequence_length, e=embedding_dim]
        # print('x_embed', x_embed.size())
        b, l, _ = x_embed.size()
        x_embed = x_embed.transpose(0, 1) # [l, b, e]
        if hidden is None:
            h_t_minus_1 = self.init_hidden(b)
        else:
            h_t_minus_1 = hidden

        output = []
        for t in range(l):
            #  Implement forward pass for a single RNN timestamp, append the hidden to the output 
            # print('x_embed: ', x_embed[t].size())
            # print('h_t_minus_1: ', h_t_minus_1.size())
            # print('self.W_h: ', self.W_h.size())
            # print('self.W_e: ', self.W_e.size())
            h_linear = torch.mm(h_t_minus_1, self.W_h)
            x_linear = torch.mm(x_embed[t], self.W_e)
            curr_hidden = torch.tanh(h_linear + x_linear) # [b, h]
            # print('curr_hidden', curr_hidden.size())
            output.append(curr_hidden)
            h_t_minus_1 = curr_hidden
        output = torch.stack(output) # [l, b, e]
        output = output.transpose(0, 1) # [b, l, e]
        

        # Set these values after completing the loop above
        # print('output[-1]: ', output[-1].size())
        final_hidden = output[:, -1, :].clone() # [b, h] 
        # print('final_hidden: ', final_hidden.size())
        logits = self.fc(output) # [b, l, vocab_size=v] 
        return logits, final_hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

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
sequence = read_file("smallwarandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))
char_to_idx = {char: idx for idx, char in enumerate(vocab)} # Create a mapping from characters to indices
idx_to_char = {idx: char for idx, char in enumerate(vocab)} # Create the reverse mapping
data = [char_to_idx[char] for char in sequence]

# Understand and adjust the hyperparameters once the code is running
sequence_length = 100 # Length of each input sequence
stride = 5            # Stride for creating sequences
embedding_dim = 200      # Dimension of character embeddings
hidden_size = 256        # Number of features in the hidden state of the RNN
learning_rate = 0.003    # Learning rate for the optimizer
num_epochs = 1        # Number of epochs to train
batch_size = 32        # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
# Split the data into 90:10 ratio with PyTorch indexing
train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]
train_dataset = CharDataset(train_data, sequence_length, stride, output_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Training loop
print('training')
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

        # compute the loss, backpropagate gradients, and update total_loss
        loss = criterion(output.view(-1, output.size(-1)), batch_targets.view(-1)) # flatten a tensor without changing its data
        loss.backward() # backpropagate
        optimizer.step() # update parameters
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
# Implement a test loop to evaluate the model on the test set
test_dataset = CharDataset(test_data, sequence_length, stride, output_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

hidden = model.init_hidden(batch_size).to(device)
total_correct = 0
total_samples = 0
total_loss = 0
print('got to testing')
with torch.no_grad():
    for batch_inputs, batch_targets in tqdm(test_loader):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        output, hidden = model(batch_inputs, hidden)
        loss = criterion(output.view(-1, output.size(-1)), batch_targets.view(-1))
        total_loss += loss.item()
        
        # Calculate accuracy
        # get the index of the maximum predicted
        _, predicted = torch.max(output, dim=2)
        total_correct += (predicted == batch_targets).sum().item() 
        total_samples += batch_targets.size(0) * batch_targets.size(1)
    print('Accuracy: ', total_correct / total_samples)

# ===================== Text Generation =====================
def sample_from_output(logits, temperature=1.0):
    """
    Sample from the logits with temperature scaling.
    logits: Tensor of shape [batch_size, vocab_size] (raw scores, before softmax)
    temperature: a float controlling the randomness (higher = more random)
    """
    if temperature <= 0:
        temperature = 0.001
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
    # readying data
    start_text = start_text.lower()
    data = [char_to_idx[char] for char in start_text]
    data_tensor = torch.tensor(data, dtype=torch.long).unsqueeze(0)  # Add batch dimension, shape: [1, n]
    generated_characters = list(start_text)

    for _ in range(k):
        output, hidden = model(data_tensor, None)
        print('output: ', output)
        # Sample the next character index from the output using the sampling function
        next_char_idx = sample_from_output(output[:, -1, :], temperature)  # Use only the last output
        next_char = idx_to_char[next_char_idx.item()]  # Extract the scalar index from the Tensor and get the char
        generated_characters.append(next_char)
        
        # Update data_tensor for the next step, which includes the newly predicted character
        data = [char_to_idx[char] for char in start_text + next_char]
        data_tensor = torch.tensor(data, dtype=torch.long).unsqueeze(0)  # Shape: [1, n + _]    
    return ''.join(generated_characters)

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