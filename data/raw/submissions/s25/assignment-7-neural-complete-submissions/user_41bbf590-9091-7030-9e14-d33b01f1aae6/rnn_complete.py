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
        self.w_e = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)
        self.w_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.w_o = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01) 


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
            #W_h*h_{t-1} + W_e*e_t
            #[b,h]
            matrix_sum = torch.matmul(h_t_minus_1, self.w_h) + torch.matmul(x_embed[t], self.w_e)
            #print(f"MS: {matrix_sum.shape}")
            #apply tanh function
            #[b,h]
            new_hidden = torch.tanh(matrix_sum)
            #print(f"NH: {new_hidden.shape}")
            #append h_t and update h_t-1
            output.append(new_hidden)
            #print(f"OP: {len(output)}")
            h_t_minus_1 = new_hidden
        output_list = output
        output = torch.stack(output) # [l, b, e]
        output = output.transpose(0, 1) # [b, l, e]
        
        # TODO set these values after completing the loop above
        final_hidden = h_t_minus_1.clone() # [b, h] 
        #g(W_o*h)
        logits_transposed = [torch.matmul(hidden,self.w_o) for hidden in output_list]# [b, l, vocab_size=v] 
        logits = torch.stack(logits_transposed).transpose(0,1)
        #print(logits.shape)
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
char_to_idx = {char: idx for (idx, char) in enumerate(vocab)} # TODO: Create a mapping from characters to indices
idx_to_char = {idx: char for (idx, char) in enumerate(vocab)} # TODO: Create the reverse mapping
data = [char_to_idx[char] for char in sequence]

# TODO Understand and adjust the hyperparameters once the code is running
sequence_length = 100 # Length of each input sequence
stride = 25             # Stride for creating sequences
embedding_dim = 3      # Dimension of character embeddings
hidden_size = 40        # Number of features in the hidden state of the RNN
learning_rate = 0.02    # Learning rate for the optimizer
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
        optimizer.zero_grad()
        #print("output shape:",output.transpose(1,2).shape)
        #print("batch targets shape:",batch_targets.shape)
        loss = criterion(output.transpose(1,2), batch_targets)
        #print(f"Loss: {loss}")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('\n')
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
# TODO: Implement a test loop to evaluate the model on the test set
test_dataset = CharDataset(test_data, sequence_length, stride, output_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
correct = 0
total = 0
with torch.no_grad():
    print()
    for X, y in test_loader:
        prediction, _ = model(X, None)
        prediction = F.softmax(prediction, dim=2)
        #reshape dimensions to do multinomial
        prediction_reshaped = torch.reshape(prediction, (-1, prediction.shape[-1]))
        prediction_reshaped = torch.multinomial(prediction_reshaped, 1)
        prediction = torch.reshape(prediction_reshaped, (prediction.shape[0], prediction.shape[1]))
        #add to correct for each correct prediction, add to total for each prediction
        #if prediction-y == 0, prediction == y, otherwise prediction-y is nonzero
        correct += torch.numel(prediction)-torch.count_nonzero(prediction-y).item()
        total += torch.numel(prediction)
print(f"Test Accuracy: {100 * correct / total:.2f}%")

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
    for i in range(k):
        model_input = torch.tensor([[char_to_idx[char] for char in start_text]], dtype=torch.long)
        logits, hidden = model(model_input, None) 
        #print(logits.shape)
        sampled_idx = sample_from_output(logits[0,:,:], temperature)
        #print(sampled_idx)
        next_char = idx_to_char[sampled_idx[-1].item()]
        #print(next_char)
        start_text += next_char

    #returns the generated text, including the original input and the newly predicted characters
    return start_text

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