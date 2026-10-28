# test_model_saving.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import os
import pytest
import torch
from unittest.mock import patch

import TicTacToeAgent
from Population import Population


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pop(device="cpu", run_id="test_run_2026"):
    """Minimal Population with one untrained candidate and a fixed run_id."""
    pop = Population(
        size=1,
        parent_count=1,
        elite_count=1,
        device=torch.device(device),
        seed=42,
        game_count=1,
    )
    # Force a deterministic run_id so filenames are predictable
    pop.run_id = run_id
    # Give the single candidate a known fitness so ranking is stable
    pop.candidates[0]["fitness"] = 0.75
    pop.candidates[0]["name"] = "TestBot"
    return pop


def _leaderboard_path():
    return Path("Models") / "leaderboard.json"


def _checkpoint_paths(run_id, generation):
    base = Path("Models") / f"champ_{run_id}_gen_{generation}"
    return base.with_suffix(".pth"), base.with_suffix(".json")


# ---------------------------------------------------------------------------
# 1. Fresh filenames + valid / missing leaderboard → save succeeds
# ---------------------------------------------------------------------------

def test_save_succeeds_with_missing_leaderboard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pop = _make_pop()

    pop.save_global_champion(
        generation=0,
        crossover=False,
        inheritance=0.5,
        starting_std=0.1,
        ending_std=0.02,
        annealing=False,
    )

    pth, meta = _checkpoint_paths(pop.run_id, 0)
    assert pth.is_file()
    assert meta.is_file()

    # Leaderboard should have been created and contain exactly one entry
    lb = _leaderboard_path()
    assert lb.is_file()
    data = json.loads(lb.read_text())
    assert len(data) == 1
    assert data[0]["filename"] == pth.name
    assert data[0]["name"] == "TestBot"
    assert data[0]["fitness"] == 0.75


def test_save_succeeds_with_valid_existing_leaderboard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Models").mkdir()

    # Pre-seed a valid leaderboard
    existing = [
        {
            "filename": "old_champ.pth",
            "generation": 5,
            "run_id": "old_run",
            "fitness": 0.60,
            "name": "OldBot",
        }
    ]
    _leaderboard_path().write_text(json.dumps(existing, indent=2))

    pop = _make_pop(run_id="new_run")
    pop.save_global_champion(0, False, 0.5, 0.1, 0.02, False)

    pth, meta = _checkpoint_paths("new_run", 0)
    assert pth.is_file()
    assert meta.is_file()

    data = json.loads(_leaderboard_path().read_text())
    assert len(data) == 2
    # New entry should be first (higher fitness)
    assert data[0]["run_id"] == "new_run"
    assert data[1]["run_id"] == "old_run"


# ---------------------------------------------------------------------------
# 2. Either checkpoint filename already exists → nothing is changed
# ---------------------------------------------------------------------------

def test_existing_pth_blocks_save(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Models").mkdir()

    pth, meta = _checkpoint_paths("test_run_2026", 0)
    pth.write_bytes(b"dummy")          # pre-create the .pth

    # Also give the leaderboard a known state
    original_lb = [{"filename": "keep_me.pth", "fitness": 0.9, "name": "Keep"}]
    _leaderboard_path().write_text(json.dumps(original_lb))

    pop = _make_pop()
    pop.save_global_champion(0, False, 0.5, 0.1, 0.02, False)

    captured = capsys.readouterr()
    assert "already Exists" in captured.out

    # Checkpoint files: the pre-existing .pth is still there, no .json was written
    assert pth.is_file()
    assert not meta.exists()

    # Leaderboard untouched
    assert json.loads(_leaderboard_path().read_text()) == original_lb


def test_existing_json_blocks_save(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Models").mkdir()

    pth, meta = _checkpoint_paths("test_run_2026", 0)
    meta.write_text("{}")              # pre-create the .json

    original_lb = [{"filename": "keep_me.pth", "fitness": 0.9, "name": "Keep"}]
    _leaderboard_path().write_text(json.dumps(original_lb))

    pop = _make_pop()
    pop.save_global_champion(0, False, 0.5, 0.1, 0.02, False)

    captured = capsys.readouterr()
    assert "already Exists" in captured.out

    assert not pth.exists()            # .pth was never written
    assert meta.is_file()              # original .json still present
    assert json.loads(_leaderboard_path().read_text()) == original_lb


# ---------------------------------------------------------------------------
# 3. Existing leaderboard contains invalid JSON → contents unchanged, failure reported
# ---------------------------------------------------------------------------

def test_invalid_leaderboard_json_leaves_file_unchanged(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Models").mkdir()

    # Write deliberately broken JSON
    broken = "{ this is not valid json"
    lb_path = _leaderboard_path()
    lb_path.write_text(broken)

    pop = _make_pop()
    pop.save_global_champion(0, False, 0.5, 0.1, 0.02, False)

    captured = capsys.readouterr()
    assert "Could not read existing leaderboard" in captured.out or \
           "Checkpoint not added to Leaderboard" in captured.out

    # The broken file must still be exactly as we left it
    assert lb_path.read_text() == broken

    # The new checkpoint files themselves *are* written (the failure is only
    # on the leaderboard update), which matches the current implementation.
    pth, meta = _checkpoint_paths(pop.run_id, 0)
    assert pth.is_file()
    assert meta.is_file()


# ---------------------------------------------------------------------------
# Extra safety: after a successful save the written .pth can be reloaded
# ---------------------------------------------------------------------------

def test_saved_checkpoint_is_reloadable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pop = _make_pop()

    pop.save_global_champion(0, False, 0.5, 0.1, 0.02, False)

    pth, _ = _checkpoint_paths(pop.run_id, 0)
    data = torch.load(pth, map_location="cpu", weights_only=False)

    assert "model_state_dict" in data
    assert "fitness" in data
    assert data["fitness"] == 0.75
    assert data["name"] == "TestBot"