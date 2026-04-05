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
        self.embedding = nn.Embedding(input_size, embedding_dim)
        # TODO: Initialize your model parameters as needed e.g. W_e, W_h, etc.
        self.W_e = nn.Parameter(torch.randn(embedding_dim, hidden_size)* 0.01)
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size)* 0.01)
        self.b_h = nn.Parameter(torch.zeros(1, hidden_size))
        self.W_o = nn.Parameter(torch.randn(hidden_size, output_size)* 0.01)
        self.b_o = nn.Parameter(torch.zeros(output_size))

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
            x_t = x_embed[t] 
            h_x = torch.matmul(x_t,self.W_e)
            h_h = torch.matmul(h_t_minus_1, self.W_h)
            h_t = torch.tanh(h_x + h_h + self.b_h)
            output.append(h_t)
            h_t_minus_1 = h_t
        output = torch.stack(output) # [l, b, e]
        output = output.transpose(0, 1) # [b, l, e]
        
        # TODO set these values after completing the loop above
        final_hidden = h_t_minus_1 # [b, h] 
        logits = torch.matmul(output,self.W_o) + self.b_o # [b, l, vocab_size=v] 
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
# sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}# TODO: Create a mapping from characters to indices
idx_to_char = {idx: char for idx, char in enumerate(vocab)} # TODO: Create the reverse mapping
data = [char_to_idx[char] for char in sequence]

# TODO Understand and adjust the hyperparameters once the code is running
sequence_length = 25 # Length of each input sequence 25 for abc, 150 for war
stride = 1            # Stride for creating sequences, 10 for war
embedding_dim = 30     # Dimension of character embeddings 30 for abc,32 for war
hidden_size = 64        # Number of features in the hidden state of the RNN 64 for abc,256 for war
learning_rate = 0.005    # Learning rate for the optimizer 0.005 for abc, 0.001 for war
num_epochs = 25      # Number of epochs to train 20 for abc, 3 for war
batch_size = 64        # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size,embedding_dim=embedding_dim).to(device)
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
    model.train()
    total_loss = 0
    hidden = None
    for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        optimizer.zero_grad()

        outputs, hidden = model(batch_inputs, hidden)
        # Detach removes the hidden state from the computational graph.
        # This is prevents backpropagating through the full history, and
        # is important for stability in training RNNs
        hidden = hidden.detach()
        
        B, L, V = outputs.size()
        loss = criterion(
            outputs.view(-1, V),          
            batch_targets.view(-1)        
        )        
        # TODO compute the loss, backpropagate gradients, and update total_loss
        
        loss.backward() 
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
# TODO: Implement a test loop to evaluate the model on the test set
model.eval()                # turn off dropout, batch-norm, etc.
test_loss = 0.0
correct   = 0
total     = 0
test_ds  = CharDataset(test_data, sequence_length, stride, vocab_size)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)
with torch.no_grad():
    hidden_test = None
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        B = inputs.size(0)
        if hidden_test is not None and hidden_test.size(0) != B:
            hidden_test = None

        outputs, hidden_test = model(inputs, hidden_test)
        hidden_test = hidden_test.detach()
        B, L, V = outputs.size()
        loss = criterion(outputs.view(-1, V),
                         targets.view(-1))
        test_loss += loss.item()
        preds   = outputs.argmax(dim=-1)   
        correct += (preds == targets).sum().item()
        total   += B * L

# average metrics
avg_test_loss = test_loss / len(test_loader)
accuracy      = correct / total * 100

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
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
    model.eval()
    hidden = None
    start_text = start_text.lower()
    generated = ''

    indices = [char_to_idx[ch] for ch in start_text]
    tensor = torch.tensor([indices], dtype=torch.long, device=device)
    outputs, hidden = model(tensor, hidden)
    last_index = indices[-1]
    for i in range(k):
        tensor = torch.tensor([[last_index]], dtype=torch.long, device=device)
        logits, hidden = model(tensor, hidden)
        hidden = hidden.detach()
        logits = logits.squeeze(1) 
        samp_index = sample_from_output(logits,temperature)
        index = samp_index.item()
        gen_char = idx_to_char[index]
        generated += gen_char
        last_index = index





    #TODO: Implement the rest of the generate_text function
    # Hint: you will call sample_from_output() to sample a character from the logits
    return generated

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