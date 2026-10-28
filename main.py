import torch
import TicTacToeAgent
import Evaluator
import GeneticOps
import Population
import json
import os
import time
from datetime import datetime
from pathlib import Path

def printEval(evaluation, game_count, device, seed):
    print("\n" + "=" * 45)
    print("         TIC-TAC-TOE AI EVALUATION")
    print("=" * 45)
    print(f" ⚙️  Configuration:")
    print(f"    • Seed:        {seed}")
    print(f"    • Device:      {device}")
    print(f"    • Game Count:  {game_count}")
    print("-" * 45)
    print(" 📊 Results:")

    for model_name, stats in evaluation.items():
        print(f"    [{model_name.upper()}]")
        print(f"      • Wins:   {stats['wins']}")
        print(f"      • Losses: {stats['losses']}")
        print(f"      • Draws:  {stats['draws']}")

    print("=" * 45)
    
def save_json(evaluation, game_count, device, seed, mutation_strength=0.1):
    save_dir = Path("Saves")
    save_dir.mkdir(exist_ok=True)
    
    base_name = "test"
    file_path = save_dir / f"{base_name}.json"
    
    counter = 1
    while file_path.exists():
        file_path = save_dir / f"{base_name}_{counter}.json"
        counter += 1
        
    info = {
        "configuration": {
            "seed": seed,
            "game_count": game_count,
            "mutation_strength": mutation_strength,
            "device": str(device),
            "pytorch_version": torch.__version__
        },
        "model_roles": {
            "model1": "Parent",
            "model2": "Child"
        },
        "results": evaluation
    }
    
    # 3. Save to the uniquely indexed path inside the 'Saves' folder
    with open(file_path, "w") as file:
        json.dump(info, file, indent=4)
        
    print(f"📁 Evaluation successfully saved to: {file_path}")

def get_champ(position, leaderboard_dir="Models"):
    """Loads a champion by its current global leaderboard rank (1 = best overall)."""
    leaderboard_path = os.path.join(leaderboard_dir, "leaderboard.json")
    
    if not os.path.exists(leaderboard_path):
        raise FileNotFoundError(f"❌ FAILURE: Leaderboard file not found at [{leaderboard_path}]. Run training first.")
        
    with open(leaderboard_path, "r") as f:
        leaderboard = json.load(f)
        
    index = position - 1
    if index >= len(leaderboard):
        raise IndexError(f"❌ FAILURE: Requested rank #{position}, but leaderboard only has {len(leaderboard)} entries.")
        
    champ_meta = leaderboard[index]
    pth_path = os.path.join(leaderboard_dir, champ_meta["filename"])
    
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"❌ FAILURE: Checkpoint file referenced in leaderboard was missing at [{pth_path}].")
    
    return pth_path

def print_champion_comparison(evaluations, names, game_count, device, seeds=None):
    """
    Pretty comparison of multiple models (champions + optional baseline).
    
    evaluations : list of evaluation lists
    names       : list of display names (must match order of evaluations)
    """
    n_models = len(evaluations)
    n_opps   = len(evaluations[0]) if n_models > 0 else 0

    if seeds is None:
        seeds = [1000 + i for i in range(n_opps)]

    print("\n" + "=" * 70)
    print("               TIC-TAC-TOE MODEL COMPARISON")
    print("=" * 70)
    print(f" ⚙️  Config:  device={device}  |  games/side={game_count}  |  opponents={n_opps}")
    print("-" * 70)

    # ---- Per-opponent breakdown ----
    print(" 📊 Results vs each opponent (wins / draws / losses | fitness)")
    print("-" * 70)

    header = f"{'Opponent':<12}"
    for name in names:
        header += f" | {name:^18}"
    print(header)
    print("-" * 70)

    for opp_idx in range(n_opps):
        row = f"seed {seeds[opp_idx]:<6}"
        for model_idx in range(n_models):
            stats = evaluations[model_idx][opp_idx]
            w, d, l = stats["wins"], stats["draws"], stats["losses"]
            fit = stats["fitness"]
            cell = f"{w:3d}/{d:3d}/{l:3d}  {fit:.3f}"
            row += f" | {cell:^18}"
        print(row)

    # ---- Aggregate summary ----
    print("-" * 70)
    print(" 📈 Aggregate (summed across all opponents)")
    print("-" * 70)

    aggregates = []
    for model_idx in range(n_models):
        total_w = sum(e["wins"]   for e in evaluations[model_idx])
        total_d = sum(e["draws"]  for e in evaluations[model_idx])
        total_l = sum(e["losses"] for e in evaluations[model_idx])
        avg_fit = sum(e["fitness"] for e in evaluations[model_idx]) / n_opps
        aggregates.append((total_w, total_d, total_l, avg_fit))

    print(f"{'Metric':<12}", end="")
    for name in names:
        print(f" | {name:^18}", end="")
    print()
    print("-" * 70)

    metrics = [
        ("Wins",    lambda a: f"{a[0]}"),
        ("Draws",   lambda a: f"{a[1]}"),
        ("Losses",  lambda a: f"{a[2]}"),
        ("Avg Fit", lambda a: f"{a[3]:.4f}"),
    ]
    for label, fmt in metrics:
        print(f"{label:<12}", end="")
        for agg in aggregates:
            print(f" | {fmt(agg):^18}", end="")
        print()

    # Ranking
    ranking = sorted(enumerate(aggregates), key=lambda x: x[1][3], reverse=True)
    print("-" * 70)
    print(" 🏆 Ranking by average fitness:")
    for rank, (idx, agg) in enumerate(ranking, 1):
        print(f"    {rank}. {names[idx]:<12}  avg fitness = {agg[3]:.4f}")
    print("=" * 70)

def evaluate_and_save_comparison(
    positions,                     # e.g. [1, 2, 3]
    device,
    opp_count=3,
    game_count=100,
    include_baseline=True,
    shouldPrint=False,
    PrintInterval=25,
    seed=1000,
    leaderboard_dir="Models",
    save_path="evaluations/champion_comparison.json"
):
    """
    Full pipeline:
    - Load champions (keeps checkpoint identity)
    - Evaluate them (+ optional baseline)
    - Print the nice table
    - Save everything to a clean JSON
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # ---------- Load champions + keep identity ----------
    champs = []
    champ_info = []
    for pos in positions:
        path = get_champ(pos, leaderboard_dir)
        checkpoint = torch.load(path, map_location=device)
        model = TicTacToeAgent.TicTacToeAI().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        info = {
            "position": pos,
            "name": checkpoint.get("name", f"champ_{pos}"),
            "generation": checkpoint.get("generation"),
            "fitness_at_save": checkpoint.get("fitness"),
            "checkpoint_path": path
        }
        champs.append(model)
        champ_info.append(info)
        print(f"Loaded {info['name']} (Gen {info['generation']}) "
              f"fitness={info['fitness_at_save']:.4f}")

    # ---------- Evaluate ----------
    evaluations = []
    names = []
    opponent_seeds = [seed + i for i in range(opp_count)]

    for i, champ in enumerate(champs):
        evals = Evaluator.evaluate_champion(
            champ, device, opp_count, seed, game_count,
            shouldPrint=shouldPrint, PrintInterval=PrintInterval
        )
        evaluations.append(evals)
        names.append(f"Champ {positions[i]}")

    baseline_seed = None
    if include_baseline:
        print("Evaluating baseline (fresh frozen model)...")
        # We use a fixed seed for reproducibility of the baseline itself
        baseline_seed = 9999
        torch.manual_seed(baseline_seed)
        baseline = TicTacToeAgent.TicTacToeAI().to(device)
        baseline.eval()

        evals = Evaluator.evaluate_champion(
            baseline, device, opp_count, seed, game_count,
            shouldPrint=shouldPrint, PrintInterval=PrintInterval
        )
        evaluations.append(evals)
        names.append("Baseline")

    # ---------- Print the nice table ----------
    print_champion_comparison(evaluations, names, game_count, device, seeds=opponent_seeds)

    # ---------- Build clean JSON ----------
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "device": str(device),
            "games_per_side": game_count,
            "total_games_per_opponent": 2 * game_count,
            "opponent_count": opp_count,
            "opponent_seeds": opponent_seeds,
            "baseline_seed": baseline_seed,
            "include_baseline": include_baseline
        },
        "champions": champ_info,
        "models": []
    }

    for model_idx, name in enumerate(names):
        model_data = {
            "display_name": name,
            "is_baseline": name == "Baseline",
            "position": None,                 # ← new
            "per_opponent": [],
            "aggregate": {
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "avg_fitness": 0.0
            }
        }

        total_w = total_d = total_l = 0.0
        for opp_idx, stats in enumerate(evaluations[model_idx]):
            model_data["per_opponent"].append({
                "opponent_seed": opponent_seeds[opp_idx],
                "wins": stats["wins"],
                "draws": stats["draws"],
                "losses": stats["losses"],
                "fitness": round(stats["fitness"], 4)
            })
            total_w += stats["wins"]
            total_d += stats["draws"]
            total_l += stats["losses"]

        model_data["aggregate"]["wins"] = int(total_w)
        model_data["aggregate"]["draws"] = int(total_d)
        model_data["aggregate"]["losses"] = int(total_l)
        model_data["aggregate"]["avg_fitness"] = round(
            sum(s["fitness"] for s in evaluations[model_idx]) / opp_count, 4
        )

        if not model_data["is_baseline"]:
            model_data["checkpoint"] = champ_info[model_idx]
            model_data["position"] = champ_info[model_idx]["position"]   # ← new

        results["models"].append(model_data)

    # Ranking
    ranked = sorted(
        results["models"],
        key=lambda m: m["aggregate"]["avg_fitness"],
        reverse=True
    )

    results["ranking"] = [
        {
            "rank": i + 1,
            "name": m["display_name"],
            "position": m["position"],                 # ← original leaderboard slot (or None)
            "avg_fitness": m["aggregate"]["avg_fitness"]
        }
        for i, m in enumerate(ranked)
    ]

    # Convenient top-level shortcuts
    best = ranked[0]
    results["current_validation_leader"] = best["display_name"]
    results["best_position"] = best["position"]        # ← what you actually need later
    results["best_is_baseline"] = best["is_baseline"]

    # ---------- Save ----------
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Comparison saved to: {save_path}")
    print(f"   Current validation leader: {results['current_validation_leader']}")
    return results


def load_champs(positions, device):
    """Load champions by their position numbers. Returns list of models."""
    champs = []
    for pos in positions:
        checkpoint = torch.load(get_champ(pos), map_location=device)
        model = TicTacToeAgent.TicTacToeAI().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"Loaded {checkpoint['name']} from Gen {checkpoint['generation']} "
              f"(fitness {checkpoint['fitness']:.3f})")
        champs.append(model)
    return champs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    seed = 10 #Version 1.0

    mutation_std = 0.1

    game_count = 100

    epochs = 10

    for i in range(epochs):
        Pop = Population.Population(size=8,
                                parent_count=2, 
                                elite_count=1, 
                                device=device, 
                                seed=seed + i,
                                mutation_std=mutation_std, 
                                game_count=game_count)
        
        Pop.run(generations=30, game_count=game_count, tournament=True, 
                tourneySize=3, annealing=True, starting_std=0.1,
                crossover=True ,inheritance=.5, printGen=False,
                saveChampion=True)
        if i == epochs - 1 or i == epochs - 2: Pop.plot_fitness()

start = time.perf_counter()

results1 = evaluate_and_save_comparison(
        positions=[1, 2, 3],
        device=device,
        opp_count=3,
        game_count=100,
        include_baseline=True,
        leaderboard_dir="data/checkpoints",
        save_path="data/comparisons/comparison_2026-10-27_1_test.json"   # or whatever you like
)

results2 = evaluate_and_save_comparison(
        positions=[results1["best_position"]],
        device=device,
        opp_count=3,
        game_count=100,
        include_baseline=True,
        seed=2000,
        leaderboard_dir="data/checkpoints",
        save_path="data/comparisons/comparison_2026-10-27_2_test.json"   # or whatever you like
)

print(f"Took {time.perf_counter() - start}s to finish")


