# --- START OF FILE rnn_complete.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np # Often useful, though maybe not strictly needed here

# ===================== Dataset =====================
class CharDataset(Dataset):
    def __init__(self, data, sequence_length, stride, vocab_size):
        """
        Args:
            data (list): List of character indices.
            sequence_length (int): The length of each input sequence.
            stride (int): The step size between the start of consecutive sequences.
            vocab_size (int): The total number of unique characters.
        """
        self.data = data
        self.sequence_length = sequence_length
        self.stride = stride
        self.vocab_size = vocab_size
        self.sequences = []
        self.targets = []

        # Create overlapping sequences with stride
        for i in range(0, len(data) - sequence_length, stride):
            # Ensure we don't go out of bounds for the target
            if i + sequence_length + 1 <= len(data):
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
        self.embedding_dim = embedding_dim
        self.output_size = output_size # Same as input_size (vocab_size)

        self.embedding = nn.Embedding(output_size, embedding_dim) # output_size is vocab_size

        # --- Implement Vanilla RNN ---
        # Using a combined Linear layer (more compact)
        self.i2h = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.h2o = nn.Linear(hidden_size, output_size) # Hidden to Output (logits)

        # Initialize weights (optional but often good practice)
        init_std = 0.01
        for param in self.parameters():
             if param.dim() > 1: # Initialize weight matrices, not biases
                 nn.init.xavier_uniform_(param) # Or kaiming, or normal_

    def forward(self, x, hidden):
        """
        x in [b, l] # b is batch_size and l is sequence length
        hidden in [b, h] # Initial or previous hidden state
        """
        x_embed = self.embedding(x)  # [b=batch_size, l=sequence_length, e=embedding_dim]
        b, l, e = x_embed.shape # Use actual shape
        x_embed = x_embed.transpose(0, 1) # [l, b, e]

        # Get the device from the model's parameters for init_hidden if needed later
        model_device = next(self.parameters()).device

        if hidden is None:
            # Initialize hidden state if not provided (start of sequence/batch)
            h_t_minus_1 = self.init_hidden(b, model_device) # Pass device
        else:
            # Ensure hidden state is on the correct device
            h_t_minus_1 = hidden.to(model_device) # Use the provided hidden state

        hidden_states = [] # Store hidden states for each time step
        for t in range(l):
            x_t = x_embed[t].to(model_device) # Ensure input time step is on the correct device [b, e]

            # --- Vanilla RNN Recurrence ---
            combined = torch.cat((x_t, h_t_minus_1), dim=1) # Concatenate along feature dim [b, e+h]
            h_t = self.activation(self.i2h(combined)) # New hidden state [b, h]

            hidden_states.append(h_t)
            h_t_minus_1 = h_t # Update the hidden state for the next time step

        output_hidden_seq = torch.stack(hidden_states) # [l, b, h]
        output_hidden_seq = output_hidden_seq.transpose(0, 1) # [b, l, h] - Batch first

        final_hidden = h_t_minus_1 # [b, h] - Note: this is h_t after the last time step t=l-1

        logits = self.h2o(output_hidden_seq) # [b, l, output_size=vocab_size]

        return logits, final_hidden

    def init_hidden(self, batch_size, device): # Added device parameter
        # Ensure hidden state is on the same device as the model parameters
        return torch.zeros(batch_size, self.hidden_size).to(device)

# ===================== Training =====================
# <<< MODIFICATION START >>>
# Check for MPS availability for M1/M2/M3 Macs
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) device.")
# Optional: Check for CUDA if you want it to work on NVIDIA GPUs too
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")
# <<< MODIFICATION END >>>

# print(f"Using device: {device}") # Redundant now

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z.,!?;:()\[\] ]+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Data Preparation ---
# sequence = "abcdefghijklmnopqrstuvwxyz" * 100 # For debugging
sequence = read_file("warandpeace.txt") # Use the text file

vocab = sorted(list(set(sequence)))
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}

data = [char_to_idx[char] for char in sequence]

# --- Hyperparameters --- (Adjust these based on experimentation)
# For War and Peace Task (Example starting point)
sequence_length = 100  # Length of each input sequence
stride = 50            # Stride for creating sequences
embedding_dim = 64     # Dimension of character embeddings (Adjusted)
hidden_size = 256      # Number of features in the hidden state (Adjusted)
learning_rate = 0.003  # Learning rate (Adjusted)
num_epochs = 1         # Number of epochs to train
batch_size = 128       # Batch size (Adjust based on memory)
# vocab_size is set above
input_size = embedding_dim # Input to RNN cell is the embedding dimension
output_size = vocab_size   # Output of RNN is prediction over vocab

# --- Model, Loss, Optimizer ---
# Ensure the model is initialized on the correct device
model = CharRNN(input_size, hidden_size, output_size, embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Data Splitting and Loading ---
data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]

print(f"Total data length: {len(data_tensor)}")
print(f"Train data length: {len(train_data)}")
print(f"Test data length: {len(test_data)}")

train_dataset = CharDataset(train_data.tolist(), sequence_length, stride, output_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print(f"Number of training batches: {len(train_loader)}")

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    hidden = None # Initialize hidden state outside the batch loop for stateful training

    for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move data to the selected device
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        optimizer.zero_grad()

        # Pass the hidden state from the previous batch
        output_logits, hidden = model(batch_inputs, hidden)

        # Detach the hidden state to prevent backpropagating through the entire history
        hidden = hidden.detach()

        # Calculate loss
        loss = criterion(output_logits.reshape(-1, output_size), batch_targets.reshape(-1))

        loss.backward()

        # Optional: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Training Loss: {average_loss:.4f}")

# Test the model
print("Evaluating on test set...")
model.eval()
test_dataset = CharDataset(test_data.tolist(), sequence_length, stride, output_size)
# Use a smaller batch size for testing if memory is an issue, but consistency is good
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

total_test_loss = 0
total_correct = 0
total_chars = 0
test_hidden = None # Reset hidden state for stateless testing

with torch.no_grad():
    for batch_inputs, batch_targets in tqdm(test_loader, desc="Testing"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        # Reset hidden state for each test batch for stateless evaluation
        test_hidden = None

        output_logits, test_hidden = model(batch_inputs, test_hidden) # test_hidden isn't really used after this

        loss = criterion(output_logits.reshape(-1, output_size), batch_targets.reshape(-1))
        total_test_loss += loss.item()

        _, predicted_indices = torch.max(output_logits, dim=2)
        correct = (predicted_indices == batch_targets).float()
        total_correct += correct.sum().item()
        total_chars += batch_targets.numel()

average_test_loss = total_test_loss / len(test_loader)
accuracy = total_correct / total_chars
print(f"Test Loss: {average_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")


# ===================== Text Generation =====================
def sample_from_output(logits, temperature=1.0):
    """
    Sample from the logits with temperature scaling.
    logits: Tensor of shape [batch_size=1, vocab_size]
    temperature: float
    """
    if temperature <= 0:
        temperature = 1e-8
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=1)
    sampled_idx = torch.multinomial(probabilities, num_samples=1)
    return sampled_idx # Shape [1, 1]

def generate_text(model, start_text, n, k, temperature=1.0):
    """
        model: Trained RNN model.
        start_text: Initial string.
        n: Length of start_text (derived).
        k: Number of additional characters to generate.
        temperature: Controls randomness.
    """
    model.eval()
    start_text = start_text.lower()

    unknown_chars = set(start_text) - set(char_to_idx.keys())
    if unknown_chars:
        print(f"Warning: Characters not in vocabulary: {unknown_chars}")
        start_text = "".join([c for c in start_text if c in char_to_idx])
        n = len(start_text)
        if n == 0:
            print("Error: Start text empty after filtering unknown characters.")
            return ""

    # Initialize hidden state on the correct device
    hidden = model.init_hidden(1, device) # Pass device

    input_indices = [char_to_idx[char] for char in start_text]
    # Move initial input to the correct device
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

    generated_text = start_text
    current_input = input_tensor

    with torch.no_grad():
        # Prime the model with the start text
        output_logits, hidden = model(current_input, hidden)
        last_logits = output_logits[:, -1, :] # Shape [1, vocab_size]

        # Generate k characters
        for _ in range(k):
            next_char_idx_tensor = sample_from_output(last_logits, temperature) # Shape [1, 1]
            next_char_idx = next_char_idx_tensor.item()

            next_char = idx_to_char[next_char_idx]
            generated_text += next_char

            # Prepare next input (the generated character) on the correct device
            current_input = next_char_idx_tensor.to(device) # Shape [1, 1]

            # Run model for one step
            output_logits, hidden = model(current_input, hidden)
            last_logits = output_logits.squeeze(1) # Shape [1, vocab_size]

    return generated_text

# ===================== Interactive Generation =====================
print("\nTraining complete. Now you can generate text.")
while True:
    try:
        start_text_input = input(f"Enter the initial text (or 'exit' to quit): ")

        if start_text_input.lower() == 'exit':
            print("Exiting...")
            break

        start_text = start_text_input
        if not start_text:
            print("Please enter some initial text.")
            continue

        n = len(start_text)
        k_input = input("Enter the number of characters to generate: ")
        if not k_input.isdigit() or int(k_input) <= 0:
            print("Please enter a positive integer for the number of characters.")
            continue
        k = int(k_input)

        temperature_input = input("Enter the temperature value (e.g., 0.8 low random, 1.0 default, 1.5 high random) [1.0]: ")
        try:
            temperature = float(temperature_input) if temperature_input else 1.0
        except ValueError:
            print("Invalid temperature value, using default 1.0.")
            temperature = 1.0

        print("\nGenerating...")
        completed_text = generate_text(model, start_text, n, k, temperature)

        print("-" * 20)
        print(f"Generated text:\n{completed_text}")
        print("-" * 20 + "\n")

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")


# --- END OF FILE rnn_complete.py ---
