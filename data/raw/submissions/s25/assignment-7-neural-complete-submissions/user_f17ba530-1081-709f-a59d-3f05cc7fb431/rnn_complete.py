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
        self.We = nn.Parameter(torch.randn(input_size, embedding_dim) * 0.01)
        self.Wh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.Wo = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)

        # Biases Do I Need This?
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.b_y = nn.Parameter(torch.zeros(output_size))

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
            #  TODO: Implement forward pass for a single RNN timestamp, append the hidden to the output 
            # x_t = x_embed[t]
            # e_t = self.We[x_t]
            # h_t = torch.tanh(h_t_minus_1 @ self.Wh.T + e_t @ torch.eye(e_t.size(-1), self.hidden_size).to(x.device) + self.b_h)
            # output.append(h_t)
            # h_t_minus_1 = h_t
            e_t = x_embed[t]  # Shape: [batch_size, embedding_dim]
            h_t = torch.tanh(h_t_minus_1 @ self.Wh.T + e_t + self.b_h)  # h_t will be [batch_size, hidden_size]
            output.append(h_t)  # Store the hidden state for this time step
            h_t_minus_1 = h_t

        output = torch.stack(output) # [l, b, e]
        output = output.transpose(0, 1) # [b, l, e]
        
        # TODO set these values after completing the loop above
        final_hidden = h_t.clone() # [b, h] 
        logits = output @ self.Wo.T + self.b_y # [b, l, vocab_size=v] 
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
sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))
char_to_idx = {char: idx for idx, char in enumerate(vocab)} # TODO: Create a mapping from characters to indices
idx_to_char = {idx: char for char, idx in char_to_idx.items()} # TODO: Create the reverse mapping
data = [char_to_idx[char] for char in sequence]

# TODO Understand and adjust the hyperparameters once the code is running
sequence_length = 200 # Length of each input sequence
stride = 20            # Stride for creating sequences
embedding_dim = 128      # Dimension of character embeddings
hidden_size = 128        # Number of features in the hidden state of the RNN
learning_rate = 0.02    # Learning rate for the optimizer
num_epochs = 1         # Number of epochs to train
batch_size = 64        # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

# added embedding_dim
model = CharRNN(input_size, hidden_size, output_size, embedding_dim=embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
#TODO: Split the data into 90:10 ratio with PyTorch indexing
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

        # TODO compute the loss, backpropagate gradients, and update total_loss
        optimizer.zero_grad()  # Zero the gradients
        # Reshape output and targets if necessary
        output = output.view(-1, output.shape[-1])  # Shape: [batch_size * sequence_length, output_size]
        batch_targets = batch_targets.view(-1)  # Shape: [batch_size * sequence_length]

        # Compute the loss
        loss = criterion(output, batch_targets)

        # Backpropagate gradients
        loss.backward()

        # Update total loss
        total_loss += loss.item()

        # Update parameters
        optimizer.step()  # Step the optimizer to update the weights


    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
# TODO: Implement a test loop to evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
test_loss = 0
correct_predictions = 0
total_predictions = 0

test_dataset = CharDataset(test_data, sequence_length, stride, output_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():  # Disable gradient computation
    for test_inputs, test_targets in test_loader:  # Assuming you have a test_loader set up
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)

        hidden = None  # Initialize hidden state
        output, hidden = model(test_inputs, hidden)

        # Reshape output and targets
        output = output.view(-1, output.shape[-1])  # Shape: [batch_size * sequence_length, output_size]
        test_targets = test_targets.view(-1)  # Shape: [batch_size * sequence_length]

        # Compute the loss for the test set
        loss = criterion(output, test_targets)
        test_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(output, dim=1)  # Get predicted class
        correct_predictions += (predicted == test_targets).sum().item()
        total_predictions += test_targets.size(0)

    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {correct_predictions / total_predictions:.4f}")


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
    #TODO: Implement the rest of the generate_text function
    # Hint: you will call sample_from_output() to sample a character from the logits
    input_indices = [char_to_idx[char] for char in start_text if char in char_to_idx]  # Convert characters to indices
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)  # Shape: [1, n]

    generated_text = start_text  # Start with the original text

    hidden = None  # Initialize hidden state
    
    for _ in range(k):
        # Get the output logits from the model
        output, hidden = model(input_tensor, hidden)  # Forward pass
        output = output[:, -1, :]  # Take the output from the last time step. Shape: [1, vocab_size]

        # Sample the next character from the output
        sampled_idx = sample_from_output(output, temperature=temperature)  # Get the index of the predicted character

        next_char_idx = sampled_idx.item()  # Get the integer index

        # Append the predicted character to the generated text
        generated_text += idx_to_char[next_char_idx]  # Append the new character

        # Update the input tensor for the next prediction
        input_tensor = torch.cat((input_tensor, sampled_idx), dim=1)
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