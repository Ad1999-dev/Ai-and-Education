import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import re
import matplotlib.pyplot as plt

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
        
        # h_t = f(W_h * h_{t-1} + W_e * e_t)
        # At the end take softmax probably

        # W_e is shape (embedding_dim, hidden_size) aka [e, h]
        self.W_e = nn.Parameter(torch.normal(0, 0.01, (embedding_dim, hidden_size)))
        # W_h is shape (hidden_size, hidden_size) aka [h, h]
        self.W_h = nn.Parameter(torch.normal(0, 0.01, (hidden_size, hidden_size)))
        # W_o is shape (hidden_size, output_size) aka [h, v]
        self.W_o = nn.Parameter(torch.normal(0, 0.01, (hidden_size, output_size)))

    def forward(self, x, hidden):
        """
        x in [b, l] # b is batch_size and l is sequence length
        """
        x_embed = self.embedding(x)  # [b=batch_size, l=sequence_length, e=embedding_dim]
        b, l, _= x_embed.size()
        x_embed = x_embed.transpose(0, 1) # [l, b, e]
        if hidden is None:
            h_t_minus_1 = self.init_hidden(b)
        else:
            h_t_minus_1 = hidden # [b, h]
        output = []
        for t in range(l):
            # TODO: Implement forward pass for a single RNN timestamp, append the hidden to the output 
            #                          [b, e]  x  [e, h]   +   [b, h]    x  [h, h]
            h_t_minus_1 = torch.tanh(x_embed[t] @ self.W_e + h_t_minus_1 @ self.W_h) # [b, h]
            output.append(h_t_minus_1)

        output = torch.stack(output) # [l, b, h]
        output = output.transpose(0, 1) # [b, l, h]
        
        # TODO set these values after completing the loop above
        final_hidden = torch.clone(h_t_minus_1) # [b, h] 
        # [b, l, h] x [h, v]
        logits = output @ self.W_o # [b, l, vocab_size=v] 
        # print("SHAPE OF LOGITS", logits.shape)
        return logits, final_hidden
    
    def init_hidden(self, batch_size):
        # [b, h]
        return torch.zeros(batch_size, self.hidden_size).to(device)

# ===================== Training =====================
# device = 'cpu'
# print(f"Using device: {device}")

device = torch.device("cpu")
# *** I found that, since the matrices in this project are relatively "small",
# *** the CPU is much faster than Metal Performance Shaders (MPS) device due
# *** to the latter's high overhead

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     testingtesting123 = torch.ones(1, device=device)
#     print(testingtesting123)
# else:
#     print("MPS device not found.")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z.,!?;:()\[\] ]+', '', text)
    return text

# To debug your model you should start with a simple sequence an RNN should predict this perfectly
# sequence = "abcdefghijklmnopqrstuvwxyz" * 100
sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))

char_to_idx = {vocab[i]: i for i in range(len(vocab))} # TODO: Create the mapping from character to index
idx_to_char = {i: vocab[i] for i in range(len(vocab))} # TODO: Create the reverse mapping

data = [char_to_idx[char] for char in sequence]

# TODO Understand and adjust the hyperparameters once the code is running
sequence_length = 100 # Length of each input sequence
stride = 10            # Stride for creating sequences
embedding_dim = 200      # Dimension of character embeddings
hidden_size = 200       # Number of features in the hidden state of the RNN
learning_rate = 0.01    # Learning rate for the optimizer
num_epochs = 1         # Number of epochs to train
batch_size = 64        # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size, embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long, device=device)
print("Data tensor shape", data_tensor.shape)
train_size = int(len(data_tensor) * 0.9)
# TODO: Split the data into 90:10 ratio with PyTorch indexing
train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]

train_dataset = CharDataset(train_data, sequence_length, stride, output_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

train_losses = []
# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    hidden = None
    count = 0
    for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        # [b, l, v] and [b, h]
        output, hidden = model(batch_inputs, hidden)
        # Detach removes the hidden state from the computational graph.
        # This is prevents backpropagating through the full history, and
        # is important for stability in training RNNs
        hidden = hidden.detach()

        # TODO: compute the loss, backpropagate gradients, and update total_loss

        output_reshaped = output.view(-1, vocab_size)
        targets_reshaped = batch_targets.view(-1)

        # print(output.shape, batch_targets.shape)
        # loss = criterion(output.reshape(batch_size, vocab_size, sequence_length), batch_targets)
        loss = criterion(output_reshaped, targets_reshaped)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
        train_losses.append(loss.detach())
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# ================Test the model==================
# TODO: Implement a test loop to evaluate the model on the test set
correct = 0
total = 0

# print(len(test_data))
test_dataset = CharDataset(test_data, sequence_length, stride, output_size)
# print(len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
# print(len(test_loader))

test_losses = []
print("Testing model...")
with torch.no_grad():
    for inputs, labels in test_loader:
        output, final_hidden = model(inputs, None)
        # print(labels.shape)
        # print(output.shape)
        correct += torch.sum(torch.argmax(output, dim=2) == labels)
        total += output.shape[1] * output.shape[0]
        
        output_reshaped = output.view(-1, vocab_size)
        targets_reshaped = labels.view(-1)
        loss = criterion(output_reshaped, targets_reshaped)
        test_losses.append(loss.detach())
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

plt.plot(train_losses)
plt.xlabel("Iteration")
plt.ylabel("Train Loss")
plt.show()

plt.plot(test_losses)
plt.xlabel("Iteration")
plt.ylabel("Test Loss")
plt.show()

try:
    print("Average train loss:", torch.mean(torch.tensor(train_losses)))
    print("Average test loss:", torch.mean(torch.tensor(test_losses)))
except:
    print("printing failed")


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
    # print(probabilities)
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

    text = start_text

    with torch.no_grad():
        for i in range(k):
            model_input = torch.tensor([char_to_idx[char] for char in text], device=device).reshape(1, -1)
            logits, _ = model(model_input, None)
            final_logits = logits[::,-1,::]
            predicted_char = idx_to_char[sample_from_output(final_logits, temperature).item()]
            text = text + predicted_char

    return text

print("Training complete. Now you can generate text.")
while True:
    start_text = input("Enter the initial text (n characters, or 'exit' or 'stop' or 'x' to quit): ")
    
    if start_text.lower() in ['exit', 'stop', 'x']:
        print("Exiting...")
        break
    
    n = len(start_text) 
    k = int(input("Enter the number of characters to generate: "))
    temperature_input = input("Enter the temperature value (1.0 is default, >1 is more random): ")
    temperature = float(temperature_input) if temperature_input else 1.0
    
    completed_text = generate_text(model, start_text, n, k, temperature)
    
    print(f"Generated text: {completed_text}")