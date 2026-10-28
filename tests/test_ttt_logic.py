import sys
from pathlib import Path

# Add the parent folder (TicTacToeRL) to Python's search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import TicTacToe

def test_win_combos():
    # Test Player 1 win (Row 0: [0, 1, 2])
    ttt = TicTacToe.TicTacToe()
    ttt.board = [1, 1, 1, 0, 0, 2, 2, 0, 0]
    assert ttt.checkWinner() == 1

    # Test Player 2 win (Column 0: [0, 3, 6])
    ttt2 = TicTacToe.TicTacToe()
    ttt2.board = [2, 1, 1, 2, 0, 0, 2, 0, 1]
    assert ttt2.checkWinner() == 2

def test_draw():
    ttt = TicTacToe.TicTacToe()
    # Full board with no winning combination
    ttt.board = [1, 2, 1, 1, 2, 2, 2, 1, 1]
    assert ttt.checkWinner() == -1
    assert ttt.Winner == -1

def test_ongoing_game():
    ttt = TicTacToe.TicTacToe()
    # Partial board with empty spots and no winner yet
    ttt.board = [1, 0, 2, 0, 0, 0, 0, 0, 0]
    assert ttt.checkWinner() == 0

def test_play_and_turn_logic():
    ttt = TicTacToe.TicTacToe()
    assert ttt.turn == 1
    
    # Valid move for Player 1 on an empty spot
    assert ttt.play(0) is True
    assert ttt.board[0] == 1
    assert ttt.turn == 2  # turn should switch to Player 2

    # Invalid move on an already occupied spot
    assert ttt.play(0) is False
    assert ttt.turn == 2  # turn should not change on an invalid move

    # Valid move for Player 2
    assert ttt.play(4) is True
    assert ttt.board[4] == 2
    assert ttt.turn == 1  # turn switches back to Player 1
    
    assert ttt.play(-1) is False
    assert ttt.turn == 1
    
    assert ttt.play(9) is False
    assert ttt.turn == 1

def test_full_game_simulation():
    """Simulates a complete game from start to finish, tracking turns and win state."""
    ttt = TicTacToe.TicTacToe()
    assert ttt.turn == 1
    assert ttt.checkWinner() == 0

    # turn 1: Player 1 plays at 0 (Top-left)
    assert ttt.play(0) is True
    assert ttt.board == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert ttt.turn == 2
    assert ttt.checkWinner() == 0

    # turn 2: Player 2 plays at 3 (Middle-left)
    assert ttt.play(3) is True
    assert ttt.board == [1, 0, 0, 2, 0, 0, 0, 0, 0]
    assert ttt.turn == 1
    assert ttt.checkWinner() == 0

    # turn 3: Player 1 plays at 1 (Top-middle)
    assert ttt.play(1) is True
    assert ttt.board == [1, 1, 0, 2, 0, 0, 0, 0, 0]
    assert ttt.turn == 2
    assert ttt.checkWinner() == 0

    # turn 4: Player 2 plays at 4 (Center)
    assert ttt.play(4) is True
    assert ttt.board == [1, 1, 0, 2, 2, 0, 0, 0, 0]
    assert ttt.turn == 1
    assert ttt.checkWinner() == 0

    # Try playing on an occupied spot (Player 1 tries to play on 4)
    assert ttt.play(4) is False
    assert ttt.turn == 1  # turn shouldn't switch on invalid move

    # turn 5: Player 1 plays at 2 (Top-right) -> Completes row [0, 1, 2] for Player 1 win!
    assert ttt.play(2) is True
    assert ttt.board == [1, 1, 1, 2, 2, 0, 0, 0, 0]
    assert ttt.Winner == 1
    
    assert ttt.play(8) is False
    assert ttt.board == [1, 1, 1, 2, 2, 0, 0, 0, 0]
    assert ttt.turn == 2
    
def test_board_conversion():
    ttt = TicTacToe.TicTacToe()
    # Test a mini board state [Empty, Player 1, Player 2]
    ttt.board = [0, 1, 2]
    converted = ttt.convert_board()
    
    expected = [
        1.0, 0.0, 0.0,  # Empty
        0.0, 1.0, 0.0,  # Player 1
        0.0, 0.0, 1.0   # Player 2
    ]
    assert converted == expected

def test_reset():
    ttt = TicTacToe.TicTacToe()
    # Modify state
    ttt.play(0)
    ttt.play(1)
    ttt.Winner = 1
    
    # Reset
    ttt.reset()
    assert ttt.board == [0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert ttt.Winner == 0
    assert ttt.turn == 1

def test_get_mask():
    ttt = TicTacToe.TicTacToe()
    ttt.board = [1, 0, 2, 0, 1, 2, 0, 1, 0]
    mask = ttt.get_mask()
    
    # 1 for empty tiles, 0 for occupied tiles
    assert mask == [0, 1, 0, 1, 0, 0, 1, 0, 1]