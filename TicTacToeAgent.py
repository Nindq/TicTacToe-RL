import torch
import torch.nn as nn
import TicTacToe
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TicTacToeAI(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden1 = nn.Linear(27, 64)
        
        self.hidden2 = nn.Linear(64, 64)
        
        self.output = nn.Linear(64, 9)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        
        return self.output(x)
    
def selectAction(model, raw_board, mask_board, deterministic=False):
    mask_tensor = torch.tensor(mask_board, dtype=torch.bool).to(device)
    
    board_tensor = torch.tensor(raw_board, dtype=torch.float32).to(device)
    board_tensor = board_tensor.reshape(1, 27)
    
    mask_tensor.to(device)
    
    logits = model(board_tensor)
    
    masked_logits = logits.masked_fill(mask_tensor == False, -1e9)
    
    if deterministic: return torch.argmax(masked_logits).item()
    
    probabilities = torch.softmax(masked_logits, dim=-1)
    
    return Categorical(probs=probabilities).sample().item()

    