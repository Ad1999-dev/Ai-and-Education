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

        # Parameters for the RNN
        self.W_e = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)  # Weight from embedding to hidden
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)    # Hidden-to-hidden weight
        self.b_h = nn.Parameter(torch.zeros(hidden_size))                         # Hidden biases

        # Parameters for the output layer
        self.W_out = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)   # Hidden-to-output weight
        self.b_out = nn.Parameter(torch.zeros(output_size))                       # Output biases


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
            # Get the current input embedding
            x_t = x_embed[t]  # [b, e]

            # Compute the new hidden state
            h_t = F.tanh(x_t @ self.W_e + h_t_minus_1 @ self.W_h + self.b_h)  # [b, h]

            # Append the new hidden state to the output list
            output.append(h_t)  # Collect hidden states

            # Update h_t_minus_1 for the next time step
            h_t_minus_1 = h_t

        output = torch.stack(output) # [l, b, e]
        output = output.transpose(0, 1) # [b, l, e]

        # TODO set these values after completing the loop above
        final_hidden = h_t.clone() # [b, h]
        logits = output @ self.W_out + self.b_out # [b, l, vocab_size=v]
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
sequence = "abcdefghijklmnopqrstuvwxyz" * 100
#sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}  # Create mapping from chars to indices
idx_to_char = {idx: char for idx, char in enumerate(vocab)}  # Create reverse mapping
data = [char_to_idx[char] for char in sequence]

# TODO Understand and adjust the hyperparameters once the code is running
sequence_length = 100 # Length of each input sequence #100, 100, 100
stride = 1            # Stride for creating sequences #3, 2, 2
embedding_dim = 64      # Dimension of character embeddings #16, 64, 128
hidden_size = 128        # Number of features in the hidden state of the RNN #64, 128, 256
learning_rate = 0.001    # Learning rate for the optimizer #0.001, 0.001, 0.0005
num_epochs = 10         # Number of epochs to train #1, 1, 1
batch_size = 64        # Batch size for training #64, 32, 64
vocab_size = len(vocab) #accuracy: 48.9%, 52.93%, 56.39%
input_size = len(vocab) #loss: 1.78, 1.58, 1.4854
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
#TODO: Split the data into 90:10 ratio with PyTorch indexing
train_data = data[:train_size]
test_data = data[train_size:]

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
        # Compute loss
        loss = criterion(output.view(-1, output_size), batch_targets.view(-1))  # Reshape for CrossEntropyLoss

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Backpropagation
        optimizer.step()       # Update weights

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
# TODO: Implement a test loop to evaluate the model on the test set
def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0  # Initialize test loss

    with torch.no_grad():  # Disable gradient tracking for inference
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to the appropriate device

            # Forward pass to get output logits
            outputs, _ = model(inputs, None)  # No need for hidden state during testing

            # Reshape outputs to [batch_size * sequence_length, vocab_size] for computing accuracy
            outputs = outputs.view(-1, outputs.size(2))  # Shape: [b * l, vocab_size]
            labels = labels.view(-1)  # Shape: [b * l]

            # Compute the test loss
            loss = criterion(outputs, labels)  # Calculate loss
            test_loss += loss.item()  # Accumulate test loss

            # Use argmax to get the predicted character indices
            _, predicted = outputs.max(dim=1)  # Get the index of the highest probability

            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)
    accuracy = round(100 * correct / total, 2)
    print(f"Test Accuracy: {accuracy}%, Test Loss: {avg_test_loss:.4f}")  # Report both test accuracy and loss

    return accuracy, avg_test_loss  # Return both accuracy and loss for further usage

# Assuming test_loader is prepared similar to train_loader with the test dataset
test_dataset = CharDataset(test_data, sequence_length, stride, output_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Call the test function
test_model(model, test_loader)

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
    model.eval()
    generated_text = start_text

    # Preprocess the start_text to the proper tensor format
    input_seq = torch.tensor([char_to_idx[char] for char in start_text], dtype=torch.long).unsqueeze(0).to(device)  # Shape: [1, n]

    hidden = None  # Initialize hidden state

    for _ in range(k):
        # Forward pass to get output logits
        output, hidden = model(input_seq, hidden)

        # Get the last output logits
        logits = output[:, -1, :]  # Shape: [1, vocab_size]

        # Sample the next character
        next_char_idx = sample_from_output(logits, temperature)

        # Append the predicted character to the generated text
        generated_text += idx_to_char[next_char_idx.item()]  # Convert index back to character

        # Prepare the next input_seq
        input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=1)  # Append the new character

        # Ensure we only keep the last n characters for prediction
        if input_seq.size(1) > n:
            input_seq = input_seq[:, 1:]  # Remove the first character to maintain the input size

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
