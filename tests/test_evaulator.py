import sys
from pathlib import Path

# Add the parent folder (TicTacToeRL) to Python's search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import TicTacToeAgent
from Evaluator import run_game

# Helper to feed a predetermined list of moves sequentially
def mock_selector_factory(move_sequence):
    move_iter = iter(move_sequence)
    def fake_select_action(model, board_rep, mask):
        return next(move_iter)
    return fake_select_action

def test_run_game_player_1_win(monkeypatch):
    # Sequence: P1 takes top row (0, 1, 2), P2 takes left column-ish (3, 4)
    moves = [0, 3, 1, 4, 2]
    monkeypatch.setattr(TicTacToeAgent, "selectAction", mock_selector_factory(moves))
    
    winner = run_game("dummy_model1", "dummy_model2")
    assert winner == 1

def test_run_game_player_2_win(monkeypatch):
    # Turn 1 (P1): 1
    # Turn 2 (P2): 3
    # Turn 3 (P1): 2
    # Turn 4 (P2): 6
    # Turn 5 (P1): 5
    # Turn 6 (P2): 0 -> Column 0 [3, 6, 0] complete for Player 2!
    moves = [1, 3, 2, 6, 5, 0]
    monkeypatch.setattr(TicTacToeAgent, "selectAction", mock_selector_factory(moves))
    
    winner = run_game("dummy_model1", "dummy_model2")
    assert winner == 2

def test_run_game_draw(monkeypatch):
    # P1: 0, P2: 1, P1: 2, P2: 4, P1: 3, P2: 5, P1: 7, P2: 6, P1: 8
    moves = [0, 1, 2, 4, 3, 5, 7, 6, 8]
    monkeypatch.setattr(TicTacToeAgent, "selectAction", mock_selector_factory(moves))
    
    winner = run_game("dummy_model1", "dummy_model2")
    assert winner == -1  # Assuming -1 represents draw

def test_run_game_rejected_move(monkeypatch):
    # Intentionally force an invalid move (e.g., trying to play on position 0 twice)
    # Turn 1: P1 plays 0 (Valid)
    # Turn 2: P2 plays 3 (Valid)
    # Turn 3: P1 plays 0 again (Invalid - position 0 is already occupied)
    moves = [0, 3, 0]
    monkeypatch.setattr(TicTacToeAgent, "selectAction", mock_selector_factory(moves))
    
    # Assert that run_game catches the failed result and raises an AssertionError
    with pytest.raises(AssertionError) as exc_info:
        run_game("dummy_model1", "dummy_model2")
        
    # Verify your custom info dictionary string is present in the failure message
    assert "'board':" in str(exc_info.value)