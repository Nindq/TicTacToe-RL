import sys
from pathlib import Path

# Add the parent folder (TicTacToeRL) to Python's search path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import torch
import copy
import pytest
import TicTacToeAgent
import Evaluator
from GeneticOps import mutation as mutate_model

def test_parent_parameters_unchanged():
    """Ensure parent model parameters do not change during mutation."""
    parent = TicTacToeAgent.TicTacToeAI()
    
    # Snapshot parent weights
    parent_weights_snapshot = [param.clone() for param in parent.parameters()]
    
    _ = mutate_model(parent, mutation_std=0.1)
    
    # Verify parent weights are completely unaltered
    for original, current in zip(parent_weights_snapshot, parent.parameters()):
        assert torch.equal(original, current), "Parent weights were modified during child mutation!"

def test_independent_child_storage():
    """Ensure child storage is deep-copied and independent from the parent."""
    parent = TicTacToeAgent.TicTacToeAI()
    child = mutate_model(parent, mutation_std=1.0)
    
    # Mutate child weights drastically or overwrite them
    with torch.no_grad():
        for param in child.parameters():
            param.fill_(999.0)
            
    # Verify parent weights do not contain the fill value
    for param in parent.parameters():
        assert not torch.equal(param, torch.full_like(param, 999.0)), "Child shares memory/reference with parent!"

def test_zero_noise_equality():
    """Mutating with zero noise should result in identical model weights."""
    parent = TicTacToeAgent.TicTacToeAI()
    child = mutate_model(parent, mutation_std=0.0)
    
    for p_param, c_param in zip(parent.parameters(), child.parameters()):
        assert torch.equal(p_param, c_param), "Zero-noise mutation altered model parameters!"

def test_positive_noise_changes():
    """Mutating with positive noise should change model weights."""
    parent = TicTacToeAgent.TicTacToeAI()
    child = mutate_model(parent, mutation_std=0.1)
    
    weights_differ = False
    for p_param, c_param in zip(parent.parameters(), child.parameters()):
        if not torch.equal(p_param, c_param):
            weights_differ = True
            break
            
    assert weights_differ, "Positive-noise mutation failed to alter any parameters!"

