import numpy as np

WIN_COMBOS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], # Columns
    [0, 4, 8], [2, 4, 6] # Diagonal
]

class TicTacToe():
    def __init__(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.Winner = 0
        self.turn = 1
    
    def play(self, pos):
        if self.Winner != 0: return False
        if pos < 0 or pos > 8: return False
        if self.board[pos] != 0: return False
        
        self.board[pos] = self.turn
        
        if self.turn == 1: 
            self.turn = 2
        else: 
            self.turn = 1
        
        self.checkWinner()
        
        return True
        
    def checkWinner(self):
        for a, b, c in WIN_COMBOS:
            if self.board[a] != 0 and self.board[a] == self.board[b] == self.board[c]:
                self.Winner = self.board[a]
                return self.board[a]
               
        if 0 not in self.board: 
            self.Winner = -1
            return -1  # draw
        return 0
    
    def reset(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.Winner = 0
        self.turn = 1
    
    def convert_board(self):
        np_board = []
        for tile in self.board:
            if tile == 0:    # Empty
                np_board.extend([1.0, 0.0, 0.0])
            elif tile == 1:  # Player X
                np_board.extend([0.0, 1.0, 0.0])
            elif tile == 2:  # Player O
                np_board.extend([0.0, 0.0, 1.0])
        
        return np_board

    def get_mask(self):
        mask_board = []
        for tile in self.board:
            if tile == 0: 
                mask_board.append(1) 
                continue
            mask_board.append(0)
        
        return mask_board