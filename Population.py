import torch
import TicTacToeAgent
import Evaluator
import GeneticOps
import namecreator
import json
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime, timezone

class Population():
    def __init__(self, size, parent_count, elite_count, device, seed=None, mutation_std=0.1, game_count=20):
        self.size = size
        self.parent_count = parent_count
        self.elite_count = elite_count
        self.mutation_std = mutation_std
        self.game_count = game_count
        self.seed = seed
        self.device = device
        
        if seed != None: torch.manual_seed(seed)
        
        self.staticModel = TicTacToeAgent.TicTacToeAI().to(device)
        self.candidates = []
        
        for i in range(size):
            candidate = {
                "name" : namecreator.generate_name(seed=(seed if i==0 else None)), 
                "fitness" : 0,
                "model" : TicTacToeAgent.TicTacToeAI().to(device),
                "wins" : 0,
                "draws" : 0,
                "losses" : 0
                        }
            self.candidates.append(candidate)
        
    def run_generation(self, game_count=False):
        for candidate in self.candidates:
            evaulation = Evaluator.multi_game(candidate["model"],
                                          self.staticModel,
                                          game_count or self.game_count,
                                          )
            result = evaulation["model1"]
            candidate["wins"] = result["wins"]
            candidate["draws"] = result["draws"]
            candidate["losses"] = result["losses"]
            candidate["fitness"] = (result["wins"] + 0.5 * result["draws"]) / (game_count or self.game_count)
         

    def record_generation(self, generation, games_played, print_summary=False, 
                        tournament=False, tourneysize=3, crossover=False,
                        inheritance=0.5, annealing=False,
                        starting_std=0.1, ending_std=0.02, max_gen=5):
            """Append this generation's stats. Call after sort, before replacing candidates."""
            if not hasattr(self, "history"):
                self.history = []
                self.run_id = datetime.now(timezone.utc).strftime("%Y_%mM_%dD__%Hh_%Mm_%Ss")

            fitnesses = [c["fitness"] for c in self.candidates]
            best = self.candidates[0] if self.candidates else None
            mean_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else None

            entry = {
                "generation": generation,
                "mutation_std": self.mutation_std if generation < (max_gen - 1) else None,
                "best_fitness": fitnesses[0] if fitnesses else None,
                "mean_fitness": mean_fitness,
                "best_name": best["name"] if best else None,
                "candidates": [
                    {
                        "rank": i,
                        "name": c["name"],
                        "fitness": c["fitness"],
                        "wins": c["wins"],
                        "draws": c["draws"],
                        "losses": c["losses"],
                    }
                    for i, c in enumerate(self.candidates)
                ],
            }
            self.history.append(entry)

            if print_summary:
                # --- TOP HEADER: Instant visibility of current generation's performance ---
                print("=" * 65)
                print(f" [GEN {generation}] Best Fitness: {entry['best_fitness']:.3f} ({entry['best_name']}) | Mean Fitness: {mean_fitness:.3f}")
                print("=" * 65)
                
                for row in entry["candidates"]:
                    print(
                        f"  {row['rank']:2d}  {row['name']:12}  "
                        f"fit {row['fitness']:.3f}  "
                        f"W {row['wins']}  D {row['draws']}  L {row['losses']}"
                    )

                # --- BOTTOM SUMMARY: Track overall progression at the end of the log block ---
                all_best_so_far = max(g["best_fitness"] for g in self.history)
                overall_mean_so_far = sum(g["mean_fitness"] for g in self.history) / len(self.history)
                print("-" * 65)
                print(f" 🎯 Run Progress -> Best Fitness So Far: {all_best_so_far:.3f} | Running Mean Fitness: {overall_mean_so_far:.3f}")
                print("-" * 65 + "\n")

            payload = {
                "config": {
                    "seed": self.seed,
                    "population_size": self.size,
                    "parent_count": self.parent_count,
                    "elite_count": self.elite_count,
                    "crossover": crossover,
                    "Inheritance_Probability": inheritance if crossover else None,
                    "mutation_std": starting_std if annealing else self.mutation_std,
                    "annealing": annealing,
                    "starting_std": starting_std if annealing else None,
                    "ending_std": ending_std if annealing else None,
                    "games_per_candidate": games_played,
                    "selection_mode": "tournament" if tournament else "truncation",
                    "tournament_size": tourneysize if tournament else None,
                    "fixed_opponent_protocol": "static untrained TicTacToeAI, model1 vs static",
                    "device": str(self.device),
                    "pytorch_version": torch.__version__,
                },
                "generations": self.history,
            }

            os.makedirs("runs", exist_ok=True)
            path = os.path.join("runs", f"run_{self.run_id}.json")
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            return path
    
    def run(self, generations=10, game_count=False, 
            tournament=False, crossover=False, tourneySize=4, annealing=False, 
            mutation_steps=None, inheritance=0.5, ending_std = 0.02, starting_std=0.1,
            printGen=False, saveChampion=False):
        
        random.seed(self.seed)
        self.mutation_std = starting_std
        if not mutation_steps: mutation_steps = generations - 1
        linearloss = 0
        if mutation_steps > 1: linearloss = (starting_std - ending_std) / (mutation_steps - 1)
        
        for gen in range(generations):
            self.run_generation(game_count)
            
            self.candidates.sort(key=lambda c: c["fitness"], reverse=True)
            self.record_generation(gen, game_count or self.game_count, 
                                   print_summary=printGen, tournament=tournament, 
                                   tourneysize=tourneySize, annealing=annealing,
                                   starting_std=starting_std, ending_std=ending_std,
                                   crossover=crossover, inheritance=inheritance,
                                   max_gen=generations)
            
            if gen == (generations - 1): break
            
            nextGeneration = []
            
            elites = self.candidates[:self.elite_count]
            nextGeneration.extend(elites)
            
            parents = GeneticOps.parentSelection(self.candidates, self.parent_count, tournament, tourneySize)
            
            for i in range(self.size - self.elite_count):
                model = None
                name = "Test"
                
                if crossover:
                    candidates = random.choices(parents, k=2)
                    model = GeneticOps.crossover(candidates[0]["model"], candidates[1]["model"], inheritance)
                    model = GeneticOps.mutation(model, self.mutation_std)
                    name = namecreator.make_child(candidates[0]["name"], candidates[1]["name"])
                else:
                    candidate = random.choice(parents)
                    model = GeneticOps.mutation(candidate["model"], self.mutation_std)
                    name = namecreator.make_child(candidate["name"])
                
                new_candidate = {
                "name" : name, 
                "fitness" : 0,
                "model" : model,
                "wins" : 0,
                "draws" : 0,
                "losses" : 0
                }
                nextGeneration.append(new_candidate)
            
            if annealing:
                self.mutation_std = max(0, self.mutation_std - linearloss)
            
            if saveChampion:
                self.save_global_champion(gen, crossover, inheritance, starting_std, ending_std, annealing)
            
            self.candidates = nextGeneration
            
        if saveChampion:
             self.save_global_champion(generations - 1, crossover, inheritance, starting_std, ending_std, annealing)
           

    
    def plot_fitness(self, show=True, save=True):
        if not getattr(self, "history", None):
            raise RuntimeError("no history; run generations with record_generation first")

        gens = [e["generation"] for e in self.history]
        best = [e["best_fitness"] for e in self.history]
        mean = [e["mean_fitness"] for e in self.history]

        plt.figure()
        plt.plot(gens, best, label="best")
        plt.plot(gens, mean, label="mean")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("fitness")
        plt.legend()
        plt.tight_layout()

        if save and hasattr(self, "run_id"):
            os.makedirs("runs", exist_ok=True)
            path = os.path.join("runs", f"run_{self.run_id}_fitness.png")
            plt.savefig(path)
        if show:
            plt.show()
        else:
            plt.close()
            
    def save_global_champion(self, generation, crossover, inheritance, starting_std, ending_std, annealing, max_global_champs=10):
        """
        Saves a permanent, uniquely named checkpoint, verifies it can reload safely, 
        and safely updates the central leaderboard using an in-memory update 
        and a verified temporary file before overwriting.
        """
        if not self.candidates:
            return
            
        os.makedirs("Models", exist_ok=True)
        best = self.candidates[0]
        
        # 1. Unique, permanent filename based on run ID and generation
        base_filename = f"Models/champ_{self.run_id}_gen_{generation}"
        pth_filename = f"{base_filename}.pth"
        json_filename = f"{base_filename}.json"
        
        config = {
            "seed": self.seed,
            "population_size": self.size,
            "parent_count": self.parent_count,
            "elite_count": self.elite_count,
            "crossover": crossover,
            "inheritance_probability": inheritance if crossover else None,
            "mutation_std": self.mutation_std,
            "annealing": annealing,
            "starting_std": starting_std if annealing else None,
            "ending_std": ending_std if annealing else None,
            "pytorch_version": str(torch.__version__),
        }
        
        checkpoint = {
            "generation": generation,
            "run_id": self.run_id,
            "fitness": best["fitness"],
            "name": best["name"],
            "model_state_dict": best["model"].state_dict(),
            "config": config
        }
        
        # 2. Save the checkpoint and its companion JSON
        try:
            torch.save(checkpoint, pth_filename)
            meta_payload = {
                "filename": os.path.basename(pth_filename),
                "generation": generation,
                "run_id": self.run_id,
                "fitness": best["fitness"],
                "name": best["name"],
                "config": config
            }
            with open(json_filename, "w") as f:
                json.dump(meta_payload, f, indent=2)
        except Exception as e:
            print(f"❌ FAILURE: Could not write checkpoint files for [{pth_filename}]. Error: {e}")
            return
            
        # 3. Verify the checkpoint can reload successfully before committing it
        try:
            verified_data = torch.load(pth_filename, map_location="cpu")
            if "model_state_dict" not in verified_data or "fitness" not in verified_data:
                raise ValueError("Loaded checkpoint dictionary is missing required keys.")
        except Exception as e:
            print(f"❌ FAILURE: Verification reload failed for file [{pth_filename}]. Error: {e}")
            if os.path.exists(pth_filename): os.remove(pth_filename)
            if os.path.exists(json_filename): os.remove(json_filename)
            return

        print(f"✅ Verified & Saved permanent checkpoint: {pth_filename} (Fitness: {best['fitness']:.3f})")
        
        # 4. Update central leaderboard safely using in-memory update + temp file verification
        leaderboard_path = os.path.join("Models", "leaderboard.json")
        temp_leaderboard_path = os.path.join("Models", "leaderboard_temp.json")
        
        # Read existing leaderboard into memory
        all_champs = []
        if os.path.exists(leaderboard_path):
            try:
                with open(leaderboard_path, "r") as f:
                    all_champs = json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Could not read existing leaderboard file [{leaderboard_path}]. Starting fresh. Error: {e}")
                
        # Update ranking in memory
        new_entry = {
            "filename": os.path.basename(pth_filename),
            "generation": generation,
            "run_id": self.run_id,
            "fitness": best["fitness"],
            "name": best["name"],
        }
        
        # Prevent exact duplicates from the same run/gen
        all_champs = [c for c in all_champs if not (c["run_id"] == self.run_id and c["generation"] == generation)]
        all_champs.append(new_entry)
        
        # Sort globally by fitness descending and truncate
        all_champs.sort(key=lambda x: x["fitness"], reverse=True)
        all_champs = all_champs[:max_global_champs]
        
        # Write updated rankings to a temporary file
        try:
            with open(temp_leaderboard_path, "w") as f:
                json.dump(all_champs, f, indent=2)
        except Exception as e:
            print(f"❌ FAILURE: Failed to write temporary leaderboard file [{temp_leaderboard_path}]. Error: {e}")
            if os.path.exists(temp_leaderboard_path): os.remove(temp_leaderboard_path)
            return
            
        # Verify the temporary file can be read correctly confirming it is not corrupt
        try:
            with open(temp_leaderboard_path, "r") as f:
                verified_leaderboard = json.load(f)
            if not isinstance(verified_leaderboard, list):
                raise ValueError("Leaderboard file content is not a valid list structure.")
        except Exception as e:
            print(f"❌ FAILURE: Verification read failed for temporary leaderboard file [{temp_leaderboard_path}]. Error: {e}")
            if os.path.exists(temp_leaderboard_path): os.remove(temp_leaderboard_path)
            return
            
        # Safely overwrite the old leaderboard file with the verified temp file
        try:
            os.replace(temp_leaderboard_path, leaderboard_path)
            print(f"🏆 Central leaderboard.json safely updated and verified!")
        except Exception as e:
            print(f"❌ FAILURE: Failed to replace old leaderboard file with temporary file. Error: {e}")
            if os.path.exists(temp_leaderboard_path): os.remove(temp_leaderboard_path)
            return