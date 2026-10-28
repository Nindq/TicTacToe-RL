# test_evaluator_aggregation_and_ops.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import random
from unittest.mock import MagicMock

import Evaluator
import GeneticOps
import TicTacToeAgent
from Population import Population


# ---------------------------------------------------------------------------
# 1. evaluate_champion aggregation (the bug you just fixed)
# ---------------------------------------------------------------------------

def _fake_multi_game_factory(champion_outcome: str):
    """
    Returns a side-effect function that produces controlled results
    for the two multi_game calls made by evaluate_champion.

    Call order inside evaluate_champion (opp_count=1):
      1. multi_game(champion, opponent, ...)  → stats come from "model1"
      2. multi_game(opponent, champion, ...)  → stats come from "model2"
    """
    call_count = {"n": 0}

    def fake_multi_game(model1, model2, game_count, shouldPrint=False, PrintInterval=25):
        call_count["n"] += 1
        n = call_count["n"]

        # Base structure
        result = {
            "model1": {"wins": 0, "draws": 0, "losses": 0},
            "model2": {"wins": 0, "draws": 0, "losses": 0},
        }

        if champion_outcome == "win":
            # First call: champion is model1 → model1 wins everything
            # Second call: champion is model2 → model2 wins everything
            if n == 1:
                result["model1"]["wins"] = game_count
                result["model2"]["losses"] = game_count
            else:
                result["model2"]["wins"] = game_count
                result["model1"]["losses"] = game_count

        elif champion_outcome == "draw":
            result["model1"]["draws"] = game_count
            result["model2"]["draws"] = game_count

        elif champion_outcome == "lose":
            # First call: champion is model1 → model1 loses everything
            # Second call: champion is model2 → model2 loses everything
            if n == 1:
                result["model1"]["losses"] = game_count
                result["model2"]["wins"] = game_count
            else:
                result["model2"]["losses"] = game_count
                result["model1"]["wins"] = game_count

        return result

    return fake_multi_game


@pytest.mark.parametrize(
    "outcome, expected_wins, expected_draws, expected_losses, expected_fitness",
    [
        ("win",  20, 0,  0, 1.0),
        ("draw", 0,  20, 0, 0.5),
        ("lose", 0,  0,  20, 0.0),
    ],
)
def test_evaluate_champion_aggregation(
    monkeypatch, outcome, expected_wins, expected_draws, expected_losses, expected_fitness
):
    """
    With one opponent and game_count=10 the champion plays 20 games total.
    The aggregation must attribute the correct side of the swap.
    """
    monkeypatch.setattr(
        Evaluator, "multi_game", _fake_multi_game_factory(outcome)
    )

    # Dummy champion – never actually used because multi_game is mocked
    champion = MagicMock()
    device = torch.device("cpu")

    results = Evaluator.evaluate_champion(
        champion, device, opp_count=1, game_count=10
    )

    assert len(results) == 1
    stats = results[0]

    assert stats["wins"] == expected_wins
    assert stats["draws"] == expected_draws
    assert stats["losses"] == expected_losses
    assert stats["wins"] + stats["draws"] + stats["losses"] == 20
    assert stats["fitness"] == pytest.approx(expected_fitness)


def test_evaluate_champion_total_games_invariant(monkeypatch):
    """Even with mixed results the total games must stay 20."""
    def mixed(model1, model2, game_count, **kwargs):
        # Always return 4/3/3 for model1 and the complement for model2
        return {
            "model1": {"wins": 4, "draws": 3, "losses": 3},
            "model2": {"wins": 3, "draws": 3, "losses": 4},
        }

    monkeypatch.setattr(Evaluator, "multi_game", mixed)

    results = Evaluator.evaluate_champion(
        MagicMock(), torch.device("cpu"), opp_count=1, game_count=10
    )
    s = results[0]
    assert s["wins"] + s["draws"] + s["losses"] == 20


# ---------------------------------------------------------------------------
# 2. Parent selection
# ---------------------------------------------------------------------------

def test_parent_selection_returns_requested_count():
    candidates = [
        {"name": f"c{i}", "fitness": i / 10.0} for i in range(20)
    ]
    parents = GeneticOps.parentSelection(
        candidates, parent_count=5, tournament=False, tourney_size=3
    )
    assert len(parents) == 5
    # Truncation should take the top 5 (already sorted by fitness desc in real use)
    # Here we just check length; ordering is the caller's responsibility.


def test_tournament_selects_highest_fitness_from_group(monkeypatch):
    """
    Force random.choices to return a known group and verify the
    highest-fitness member is chosen.
    """
    candidates = [
        {"name": "low",  "fitness": 0.1},
        {"name": "mid",  "fitness": 0.5},
        {"name": "high", "fitness": 0.9},
        {"name": "other","fitness": 0.3},
    ]

    # Always sample the first three
    monkeypatch.setattr(
        random, "choices",
        lambda population, k: [candidates[0], candidates[1], candidates[2]]
    )

    parents = GeneticOps.parentSelection(
        candidates, parent_count=1, tournament=True, tourney_size=3
    )

    assert len(parents) == 1
    assert parents[0]["name"] == "high"
    assert parents[0]["fitness"] == 0.9


def test_tournament_repeated_selection_still_returns_count():
    candidates = [{"name": f"c{i}", "fitness": random.random()} for i in range(15)]
    parents = GeneticOps.parentSelection(
        candidates, parent_count=7, tournament=True, tourney_size=4
    )
    assert len(parents) == 7


# ---------------------------------------------------------------------------
# 3. Annealing / reproduction-strength schedule
# ---------------------------------------------------------------------------

def test_annealing_schedule_six_generations():
    """
    six evaluated generations → five reproduction steps with
    strengths .10, .08, .06, .04, .02; the final generation produces no children.
    """
    device = torch.device("cpu")
    pop = Population(
        size=4,
        parent_count=2,
        elite_count=1,
        device=device,
        seed=42,
        mutation_std=0.1,
        game_count=2,
    )

    # Spy on the mutation std that is actually used for each child
    recorded_stds = []

    original_mutation = GeneticOps.mutation

    def tracking_mutation(model, std):
        recorded_stds.append(std)
        return original_mutation(model, std)

    # We also need to prevent the real multi_game from running
    # (it would be slow and non-deterministic for this unit test)
    def dummy_multi_game(model1, model2, game_count, **kwargs):
        return {
            "model1": {"wins": 1, "draws": 0, "losses": game_count - 1},
            "model2": {"wins": game_count - 1, "draws": 0, "losses": 1},
        }

    import Evaluator as Ev
    original_mg = Ev.multi_game
    Ev.multi_game = dummy_multi_game
    GeneticOps.mutation = tracking_mutation

    try:
        pop.run(
            generations=6,
            game_count=2,
            annealing=True,
            starting_std=0.10,
            ending_std=0.02,
            mutation_steps=5,          # five reproduction steps
            printGen=False,
            saveChampion=False,
        )
    finally:
        Ev.multi_game = original_mg
        GeneticOps.mutation = original_mutation

    # Expected linear schedule: 0.10 → 0.08 → 0.06 → 0.04 → 0.02
    expected = [0.10, 0.08, 0.06, 0.04, 0.02]

    # Each generation (except the last) creates (size - elite_count) children
    children_per_gen = pop.size - pop.elite_count   # 3
    assert len(recorded_stds) == 5 * children_per_gen

    # Group by generation and check the std that was used
    for gen_idx, exp_std in enumerate(expected):
        start = gen_idx * children_per_gen
        end = start + children_per_gen
        gen_stds = recorded_stds[start:end]
        assert all(pytest.approx(s, abs=1e-6) == exp_std for s in gen_stds), \
            f"Generation {gen_idx} used {gen_stds}, expected {exp_std}"


# ---------------------------------------------------------------------------
# 4. Action masking – both selection modes avoid occupied squares
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _occupied_mask_and_board():
    """
    mask_board: 1 = legal, 0 = occupied.
    Positions 0, 2, 4 are illegal.
    raw_board is a flat 27-element vector (one-hot style that the network expects).
    """
    mask = [0, 1, 0, 1, 0, 1, 1, 1, 1]          # 0,2,4 occupied
    # Dummy 27-dim board (content does not matter for the mask test)
    board = [0.0] * 27
    return board, mask


def test_select_action_avoids_occupied_deterministic():
    board, mask = _occupied_mask_and_board()

    class FakeNet(torch.nn.Module):
        def forward(self, x):
            # Put the highest score on an illegal square (index 0)
            return torch.tensor(
                [[30.0, 1.0, 20.0, 2.0, 25.0, 3.0, 4.0, 5.0, 6.0]],
                device=x.device          # ← critical line
            )

    model = FakeNet().to(device)
    model.eval()

    action = TicTacToeAgent.selectAction(model, board, mask, deterministic=True)

    assert mask[action] == 1, f"Deterministic path selected illegal square {action}"
    assert action not in (0, 2, 4)


def test_select_action_avoids_occupied_stochastic():
    """
    Stochastic sampling must also respect the mask.
    Run many times; never land on an illegal square.
    """
    board, mask = _occupied_mask_and_board()

    class FakeNet(torch.nn.Module):
        def forward(self, x):
            # Massive probability mass on the three illegal squares
            return torch.tensor([[40.0, 0.01, 35.0, 0.01, 38.0, 0.01, 0.01, 0.01, 0.01]],
                                device=x.device)

    model = FakeNet().to(device)
    model.eval()

    illegal_hits = set()
    for _ in range(100):
        action = TicTacToeAgent.selectAction(model, board, mask, deterministic=False)
        if mask[action] == 0:
            illegal_hits.add(action)

    assert not illegal_hits, f"Stochastic path selected illegal squares: {illegal_hits}"


def test_select_action_prefers_legal_when_scores_close():
    """Sanity: when the best legal square is clearly better, both modes pick it."""
    board, mask = _occupied_mask_and_board()

    class FakeNet(torch.nn.Module):
        def forward(self, x):
            # Best legal square is index 7
            return torch.tensor([[1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 5.0, 50.0, 6.0]],
                                device=x.device)

    model = FakeNet().to(device)
    model.eval()

    det = TicTacToeAgent.selectAction(model, board, mask, deterministic=True)
    assert det == 7

    # Stochastic should still land on 7 most of the time
    samples = [
        TicTacToeAgent.selectAction(model, board, mask, deterministic=False)
        for _ in range(30)
    ]
    assert samples.count(7) > 20          # very high probability
    assert all(mask[a] == 1 for a in samples)

# ---------------------------------------------------------------------------
# Convenience: run only the new aggregation tests quickly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "evaluate_champion or parent_selection or tournament or annealing or select_action"])