import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np # Keep import for consistency, though not used in M1

# ===================== Dataset =====================
class CharDataset(Dataset):
    def __init__(self, data, sequence_length, stride, vocab_size):
        self.data = data
        self.sequence_length = sequence_length
        self.stride = stride
        self.vocab_size = vocab_size
        self.sequences = []
        self.targets = []

        # Ensure data is long enough before creating sequences
        if len(data) <= sequence_length:
             print(f"Warning: Data length ({len(data)}) is less than or equal to sequence length ({sequence_length}). Adjust parameters or data.")
             # Handle edge case if possible
             if len(data) > 1:
                 self.sequences.append(data[:-1])
                 self.targets.append(data[1:])
        else:
            # Create overlapping sequences with stride
            for i in range(0, len(data) - sequence_length, stride):
                # stay within the data
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
    # Using input_size consistently for vocab size dimensions
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=30):
        super().__init__()
        # Ensure input_size and output_size are the same, representing vocab size
        assert input_size == output_size, \
            "For this CharRNN, input_size and output_size must be the same (vocab size)."

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = input_size # Use input_size internally for vocab size

        # Embedding layer uses vocab_size (from input_size) and embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Manual RNN Parameters
        self.W_xh = nn.Parameter(torch.randn(self.embedding_dim, self.hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(self.hidden_size))

        # Output layer maps hidden_size to vocab_size (from input_size)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, hidden):
        """
        Manually implements the forward pass for a basic RNN cell over sequence length.
        x shape: [batch_size, sequence_length] (indices up to vocab_size)
        hidden shape: [batch_size, hidden_size] (Assumes num_layers=1)
        """
        b, l = x.size()
        x_embed = self.embedding(x) # Shape: [b, l, embedding_dim]

        if hidden is None:
            h_t_minus_1 = self.init_hidden(b) # Shape: [b, hidden_size]
        else:
            if hidden.dim() == 3 and hidden.size(0) == 1:
                 h_t_minus_1 = hidden.squeeze(0)
            elif hidden.dim() == 2 and hidden.size() == (b, self.hidden_size):
                 h_t_minus_1 = hidden
            else:
                 raise ValueError(f"Manual RNN expects hidden shape [batch_size, hidden_size] or [1, batch_size, hidden_size], got {hidden.shape}")

        output_hidden_states = []

        for t in range(l):
            x_t = x_embed[:, t, :] # Shape: [b, embedding_dim]
            h_t = torch.tanh(torch.matmul(x_t, self.W_xh) + torch.matmul(h_t_minus_1, self.W_hh) + self.b_h)
            output_hidden_states.append(h_t)
            h_t_minus_1 = h_t

        output = torch.stack(output_hidden_states, dim=1) # Shape: [b, l, hidden_size]
        final_hidden = h_t.clone() # Shape: [b, hidden_size]
        logits = self.fc(output) # Shape: [b, l, vocab_size]

        return logits, final_hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state tensor."""
        weight = next(self.parameters()).data
        # Shape: [batch_size, hidden_size]
        return torch.zeros(batch_size, self.hidden_size, device=weight.device)

# ===================== Training =====================
# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r"[^a-z.,!?;:()\[\]\s]+", '', text) # Keep letters, basic punctuation, space
        text = re.sub(r'\s+', ' ', text).strip()
    return text

#sequence = "abcdefghijklmnopqrstuvwxyz" * 100
sequence = read_file("warandpeace.txt") 

if not sequence:
    raise ValueError("Input sequence is empty.")

vocab = sorted(list(set(sequence)))
if not vocab:
     raise ValueError("Vocabulary is empty.")

char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}

data = [char_to_idx[char] for char in sequence]

# --- Hyperparameters ---
sequence_length = 50  # Length of sequence fed to RNN
stride = 10              # Stride for dataset creation 
embedding_dim = 15      # Dimension of character embeddings 
hidden_size = 128        # RNN hidden state size 
num_layers = 1           # Number of RNN layers 
learning_rate = 0.001     # Learning rate 
num_epochs = 1          # Number of epochs 
batch_size = 128        # Batch size

vocab_size = len(vocab)
input_size = vocab_size  
output_size = vocab_size 

# --- initializing the model ---
model = CharRNN(input_size, hidden_size, output_size, embedding_dim=embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Using Adam

# --- Data Splitting and Loaders ---
data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)

# TODO: Split the data into 90:10 ratio with PyTorch indexing 
train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]

# Check for empty datasets after split
if len(train_data) == 0 or len(test_data) == 0:
    raise ValueError("Train or test data is empty after splitting. Check data length and split ratio.")

train_dataset = CharDataset(train_data.tolist(), sequence_length, stride, vocab_size)
if len(train_dataset) == 0:
     raise ValueError(f"Training dataset has length 0. Check data length ({len(train_data)}), sequence length ({sequence_length}), and stride ({stride}).")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# ===================== Training Loop =====================
print("Starting training...")
model.train() # Set model to training mode
for epoch in range(num_epochs):
    total_loss = 0
    hidden = None # Reset hidden state at the start of each epoch for stateless training

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_inputs, batch_targets in pbar:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        # 1. Zero gradients, calculate loss, Backward pass, and Gradient Clipping 
        optimizer.zero_grad()
        output, hidden = model(batch_inputs, hidden)
        loss = criterion(output.reshape(-1, vocab_size), batch_targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.detach()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Average Training Loss: {avg_loss:.4f}")

# ===================== Test Loop =====================
# TODO: Implement a test loop 
print("\nStarting testing...")
model.eval() # Set model to evaluation mode

test_dataset = CharDataset(test_data.tolist(), sequence_length, stride, vocab_size)
if len(test_dataset) == 0:
    print(f"Warning: Test dataset has length 0. Skipping testing.")
    test_loss = float('nan')
    avg_test_loss = float('nan')
    test_accuracy = float('nan') # Accuracy is also not calculable

else:
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    total_test_loss = 0
    test_hidden = None # Initialize hidden state for testing

    total_correct_predictions = 0
    total_predictions_count = 0

    with torch.no_grad(): # Disable gradient calculations
        for batch_inputs, batch_targets in tqdm(test_loader, desc="Testing"):
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            # Forward pass
            # output shape: [batch_size, sequence_length, vocab_size]
            output, test_hidden = model(batch_inputs, test_hidden)

            #Loss Calculation
            loss = criterion(output.reshape(-1, vocab_size), batch_targets.reshape(-1))
            total_test_loss += loss.item()

            # Accuracy Calculation
            predictions = torch.argmax(output, dim=-1) # Use dim=-1 for the last dimension
            correct_predictions_tensor = (predictions == batch_targets)
            total_correct_predictions += correct_predictions_tensor.sum().item() 
            total_predictions_count += batch_targets.numel() 
            # Detach hidden state if stateful (less critical in eval, but consistent)
            if isinstance(test_hidden, torch.Tensor):
                test_hidden = test_hidden.detach()
            elif isinstance(test_hidden, tuple):
                 test_hidden = tuple(h.detach() for h in test_hidden)

    if len(test_loader) > 0:
        avg_test_loss = total_test_loss / len(test_loader)
        # Calculate accuracy as a percentage
        test_accuracy = (total_correct_predictions / total_predictions_count) * 100
        print(f"Average Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
    else:
        print("Test loader was empty, cannot calculate test loss.")
        avg_test_loss = float('nan') # Or handle as appropriate

# ===================== Text Generation =====================
def sample_from_output(logits, temperature=1.0):
    # Keeping the provided implementation, assuming it's okay for M1 context
    if temperature <= 0:
        print("Warning: Temperature must be positive. Using temperature=1e-8.")
        temperature = 1e-8
    if logits.dim() > 2: logits = logits.squeeze()
    if logits.dim() == 1: logits = logits.unsqueeze(0)
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=1)
    sampled_idx = torch.multinomial(probabilities, 1)
    return sampled_idx.squeeze()

def generate_text(model, start_text, n, k, temperature=1.0):
    """Generates k characters following the start_text."""
    model.eval()
    generated_text = start_text.lower()
    start_text_processed = generated_text
    # Filter unknown chars
    unknown_chars = set(start_text_processed) - set(char_to_idx.keys())
    if unknown_chars: start_text_processed = "".join([ch for ch in start_text_processed if ch in char_to_idx])
    if not start_text_processed: return start_text # Return original if invalid start

    input_indices = [char_to_idx[ch] for ch in start_text_processed]
    input_seq = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
    generated_chars = []
    hidden = None

    with torch.no_grad():
        # Process start_text
        output, hidden = model(input_seq, hidden)
        next_char_logits = output[:, -1, :]
        # Generate k chars
        for _ in range(k):
            next_char_idx = sample_from_output(next_char_logits, temperature)
            generated_chars.append(idx_to_char[next_char_idx.item()])
            current_input = next_char_idx.unsqueeze(0).unsqueeze(0)
            output, hidden = model(current_input, hidden)
            next_char_logits = output.squeeze().unsqueeze(0)
    return generated_text + "".join(generated_chars)

# ===================== Interactive Loop =====================
print("\nTraining complete")
while True:
    try:
        start_text = input(f"Enter the initial text (e.g., 'abc', or 'exit' to quit): ")
        if start_text.lower() == 'exit':
            print("Exiting...")
            break
        if not start_text:
            print("Start text cannot be empty.")
            continue

        n = len(start_text)

        k_input = input("Enter the number of characters to generate (e.g., 10): ")
        k = int(k_input)

        temp_input = input("Enter the temperature (e.g., 0.8. Default is 1.0): ")
        temperature = float(temp_input) if temp_input else 1.0

        print("\nGenerating text...")
        completed_text = generate_text(model, start_text, n, k, temperature) # Calls the unimplemented function

        print("-" * 20)
        print(f"Output: {completed_text}")
        print("-" * 20)

    except ValueError as e:
        print(f"Invalid input error: '{e}'. Please ensure you enter valid numbers. Please try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
