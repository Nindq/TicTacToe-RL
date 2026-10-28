import torch
import TicTacToeAgent
import copy
import random
import numpy as np

def mutation(parentModel, mutation_std=0.1):
    childModel = copy.deepcopy(parentModel)
    
    with torch.no_grad():
        for param in childModel.parameters():
            
            noise = torch.randn_like(param) * mutation_std
            
            param.add_(noise)
    
    return childModel

def parentSelection(candidates, parent_count, tournament, tourney_size):
    if not tournament: return candidates[:parent_count]
    
    parents = []
    
    while (len(parents) < parent_count):
        participants = random.choices(candidates, k=tourney_size)
        participants.sort(key=lambda c: c["fitness"], reverse=True)
        parents.append(participants[:1][0])
    
    return parents

def crossover(parentA, parentB, inheritance):
    childModel = copy.deepcopy(parentA)
    
    if inheritance <= 0: return childModel
    if inheritance >= 1: return copy.deepcopy(parentB)
    named_params = {}
    with torch.no_grad():
        for name, param in parentA.named_parameters(): 
            if not named_params.get(name): named_params[name] = []
            named_params[name].append(param)
            
        for name, param in parentB.named_parameters(): 
            if not named_params.get(name): named_params[name] = []
            named_params[name].append(param)
            
        for name, param in childModel.named_parameters():
            chance = torch.rand_like(param)
            param.copy_(torch.where(chance <= inheritance,
                                    named_params[name][1],
                                    named_params[name][0]))
        
    return childModel
            
            