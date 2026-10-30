import json
import os
from matplotlib import pyplot as plt
from pathlib import Path

run_path = "data/training/run_2026_10M_28D__02h_31m_26s.json"

with open(run_path, "r") as f:
    data = json.load(f)

generations = data["generations"]

best_fitness = [gen["best_fitness"] for gen in generations]
mean_fitness = [gen["mean_fitness"] for gen in generations]
gen_list = [gen["generation"] for gen in generations]

plt.plot(gen_list, mean_fitness, label='Population Mean')
plt.plot(gen_list, best_fitness, label='Generation Best')

plt.annotate('Champion Model\n(Selected Checkpoint)', xy=(28, .75), xytext=(23, .78),
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.ylabel("Fitness Score")
plt.xlabel("Generations")
plt.title(f"Training fitness seed - {data["config"]["seed"]}")

plt.legend()

os.makedirs("data/graphs", exist_ok=True)
path = os.path.join("data/graphs", f"Training_fitness_seed-{data["config"]["seed"]}.png")
plt.savefig(path)
plt.show()