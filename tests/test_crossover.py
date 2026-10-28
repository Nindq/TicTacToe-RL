import sys
from pathlib import Path

# Add the parent folder (TicTacToeRL) to Python's search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import copy
import pytest
import TicTacToeAgent
from GeneticOps import crossover

def test_crossover_inheritance_zero():
    """Ensure crossover with inheritance <= 0 returns a model identical to parentA."""
    parentA = TicTacToeAgent.TicTacToeAI()
    parentB = TicTacToeAgent.TicTacToeAI()
    
    # Modify parentB weights so they are distinct from parentA
    with torch.no_grad():
        for param in parentB.parameters():
            param.fill_(999.0)
            
    child = crossover(parentA, parentB, inheritance=0.0)
    
    for a_param, c_param in zip(parentA.parameters(), child.parameters()):
        assert torch.equal(a_param, c_param), "Crossover with inheritance=0.0 failed to match parentA!"

def test_crossover_inheritance_one():
    """Ensure crossover with inheritance >= 1 returns a model identical to parentB."""
    parentA = TicTacToeAgent.TicTacToeAI()
    parentB = TicTacToeAgent.TicTacToeAI()
    
    # Modify parentB weights so they are distinct from parentA
    with torch.no_grad():
        for param in parentB.parameters():
            param.fill_(999.0)
            
    child = crossover(parentA, parentB, inheritance=1.0)
    
    for b_param, c_param in zip(parentB.parameters(), child.parameters()):
        assert torch.equal(b_param, c_param), "Crossover with inheritance=1.0 failed to match parentB!"

def test_parents_parameters_unchanged():
    """Ensure parentA and parentB parameters do not change during crossover."""
    parentA = TicTacToeAgent.TicTacToeAI()
    parentB = TicTacToeAgent.TicTacToeAI()
    
    # Snapshot parent weights
    parentA_snapshot = [param.clone() for param in parentA.parameters()]
    parentB_snapshot = [param.clone() for param in parentB.parameters()]
    
    _ = crossover(parentA, parentB, inheritance=0.5)
    
    # Verify parentA weights are completely unaltered
    for original, current in zip(parentA_snapshot, parentA.parameters()):
        assert torch.equal(original, current), "ParentA weights were modified during crossover!"
        
    # Verify parentB weights are completely unaltered
    for original, current in zip(parentB_snapshot, parentB.parameters()):
        assert torch.equal(original, current), "ParentB weights were modified during crossover!"

def test_crossover_intermediate_mixing():
    """Ensure intermediate inheritance blends weights from both parents correctly."""
    parentA = TicTacToeAgent.TicTacToeAI()
    parentB = TicTacToeAgent.TicTacToeAI()
    
    # Assign distinct, uniform weights to each parent
    with torch.no_grad():
        for param in parentA.parameters():
            param.fill_(1.0)
        for param in parentB.parameters():
            param.fill_(2.0)
            
    # Fix random seed to make the stochastic crossover deterministic for testing
    torch.manual_seed(42)
    child = crossover(parentA, parentB, inheritance=0.5)
    
    has_from_a = False
    has_from_b = False
    
    for p_a, p_b, c in zip(parentA.parameters(), parentB.parameters(), child.parameters()):
        # 1. Verify every single element in the child comes from EITHER parentA or parentB
        valid_genes = (c == p_a) | (c == p_b)
        assert valid_genes.all(), "Child contains weights that didn't come from parentA or parentB!"
        
        # 2. Track whether we actually see both sources represented
        if (c == p_a).any():
            has_from_a = True
        if (c == p_b).any():
            has_from_b = True
            
    # Ensure the child is a true hybrid (contains traits from both parents)
    assert has_from_a, "Child has no traits from parentA!"
    assert has_from_b, "Child has no traits from parentB!"