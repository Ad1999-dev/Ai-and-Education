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
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=30): #if em
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_dim) 
        # TODO: Initialize your model parameters as needed e.g. W_e, W_h, etc.
        self.W_e = nn.Parameter(torch.randn(embedding_dim, hidden_size)) 
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_o = nn.Parameter(torch.rand(hidden_size,output_size)) 

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
            #tanh function of hidden * w_h + learnable vector * w_e
            x_t = x_embed[t, :, :]
            # print(h_t_minus_1.shape, self.W_h.shape)
            # print(x_t.shape, self.W_e.shape)
            h_t_minus_1 = torch.tanh(torch.mm(h_t_minus_1, self.W_h) + torch.mm(x_t, self.W_e))
            output.append(h_t_minus_1)
            
        output = torch.stack(output) # [l, b, e]
        output = output.transpose(0, 1) # [b, l, e]
        
        # TODO set these values after completing the loop above
        final_hidden = h_t_minus_1.clone().detach()
        # [b, h] 
        logits = torch.matmul(output, self.W_o)
        # [b, l, vocab_size=v] 
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
#sequence = "abcdefghijklmnopqrstuvwxyz" * 100
sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))
char_to_idx = {} # TODO: Create a mapping from characters to indices, indices where they occur in the sequence or in vocab
#mapping to indexes in vocab
idx_to_char = {} # TODO: Create the reverse mapping
for idx, char in enumerate(vocab):
    char_to_idx[char] = idx
    idx_to_char[idx] = char

data = [char_to_idx[char] for char in sequence]

# TODO Understand and adjust the hyperparameters once the code is running
sequence_length = 36 # Length of each input sequence
stride = 2           # Stride for creating sequences
embedding_dim = 6      # Dimension of character embeddings
hidden_size =  18       # Number of features in the hidden state of the RNN
learning_rate = 0.01  # Learning rate for the optimizer
num_epochs = 1         # Number of epochs to train
batch_size = 64        # Batch size for training
vocab_size = len(vocab)
#print("vocab size: ", vocab_size)
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    hidden = None
    for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        
        #forward pass
        logits, hidden = model(batch_inputs, hidden)
        # Detach removes the hidden state from the computational graph.
        # This is prevents backpropagating through the full history, and
        # is important for stability in training RNNs
        hidden = hidden.detach()

        # TODO compute the loss, backpropagate gradients, and update total_loss
        
        #compute loss
        logits = logits.view(-1, output_size)
        batch_targets = batch_targets.view(-1)

        loss = criterion(logits, batch_targets)
        total_loss += loss.item()
        
        #backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    total_loss = total_loss/len(train_loader)

    print(f"Epoch {epoch+1}, Loss: {total_loss:.3f}")

# Test the model
# TODO: Implement a test loop to evaluate the model on the test set
test_dataset = CharDataset(test_data, 100, stride, output_size)
#print("test dataset len: ", len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

correct = 0
total = 0

model.eval()
with torch.no_grad():
    #hidden = None 
    
    for batch_inputs, batch_targets in tqdm(test_loader):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        hidden = None
        output, hidden = model(batch_inputs, hidden)
        hidden = hidden.detach() #need to remove hidden state in test? 
    
        output = output.view(-1, output_size)
        batch_targets = batch_targets.view(-1)

        _, predicted = torch.max(output, 1)
        total += batch_targets.size(0)
        correct += (predicted == batch_targets).sum().item()
        
    accuracy = (correct / total) * 100
print(f'Accuracy of the model on the test set: {accuracy:.4f}')
                        
            

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
    #convert start text to idx
    model.eval()
    hidden = None
    
    #converting strings to tensor
    input_idxs = [char_to_idx[char] for char in start_text]
    input_tensor = torch.tensor(input_idxs, dtype=torch.long).to(device)
    input_tensor = input_tensor.view(1, -1)
    #print("input tensor shape: ", input_tensor.shape)
    
    generated_text = start_text
    current_input = input_tensor[:, -1:]
   # print("current input shape: ", current_input.shape)
    for _ in range(k):
        
        logits, _ = model(current_input, hidden)
        logits = logits[:, -1, :]
        next_idx = sample_from_output(logits, temperature)
        #convert output idx to char
        next_char = idx_to_char[next_idx.item()]
        generated_text += next_char
        
        #prepare input for the next time step
        current_input = next_idx.view(1, 1)

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