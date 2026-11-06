# Data and Evidence Index

This directory contains the experimental runs, trained model checkpoints, evaluation reports, and generation histories for the Tic-Tac-Toe neuroevolution project. Its purpose is to link the results summarized here to their saved source records.

## Evidence Kept and Verification

This repository retains records for checking the reported training curves, listed hyperparameter comparisons, and evaluation scores. Older logs omit some configuration fields, so these records do not guarantee exact reproduction of every historical training run:

- **Full training logs:** Complete 30-generation records for the final training batch (seeds 10, 11, and 12), logging the fitness of every candidate in every generation.
- **Model weights and metadata:** Saved PyTorch weights (`.pth`) and sidecar metadata (`.json`) for top candidates, indexed by a leaderboard.
- **Evaluation reports:** Head-to-head performance benchmarks against held-out opponent pools and a seeded untrained baseline across hundreds of balanced games.
- **Ablation experiment records:** 27 indexed historical run logs supporting matched-seed comparisons of selection methods, mutation schedules, and crossover. The folders also contain duplicated copies and additional archived runs; the tables below identify the comparison set.
- **Visual figures:** The generated training curve plot for seed 12.

With these records, anyone can verify:
1. The generation-by-generation fitness curves without spending hours re-running training simulations.
2. Why the generation-28 model from seed 12 was selected as champion over generation 16 (validation score of 0.6650 versus 0.6008 on held-out opponents).
3. The exact final-test statistics (370 wins, 70 draws, 160 losses across 600 games against unseen opponents, yielding a 0.6750 fitness score).
4. The recorded settings, seeds, and outcomes of the indexed historical experiments, separating measured observations from interpretations. Some settings described below come from the historical code review rather than explicit fields in every JSON.

## Folder Structure

The `data/` folder is divided into five subdirectories:

- `checkpoints/`: Stores saved neural network weights (`.pth`) and companion metadata (`.json`) for top-performing candidates, indexed by [leaderboard.json](checkpoints/leaderboard.json).
- `comparisons/`: Contains structured evaluation reports ([comparison_2026-10-27_1.json](comparisons/comparison_2026-10-27_1.json) and [comparison_2026-10-27_2.json](comparisons/comparison_2026-10-27_2.json)) measuring candidate models against held-out, untrained opponents outside training.
- `training/`: Contains full generation histories for completed 30-generation training runs across seeds 10, 11, and 12.
- `graphs/`: Contains visual figures, specifically [Training_fitness_seed-12.png](graphs/Training_fitness_seed-12.png), illustrating population mean and best candidate fitness over 30 generations.
- `experiments/`: Contains raw run logs from controlled ablation experiments organized into `selections/`, `annealing/`, and `crossover/`.

### Distinguishing Data Types

- **Model weights** (PyTorch `.pth` files in `checkpoints/`): Serialized binary state dictionaries containing the actual learned floating-point weights and biases of the neural network.
- **Measured results** (JSON files in `training/`, `comparisons/`, and `experiments/`): Structured quantitative records containing raw game outcomes (wins, draws, losses), calculated fitness scores, generation statistics, and exact run configurations.
- **Visualizations** (PNG image files in `graphs/`): Visual plots generated directly from recorded quantitative data to show performance trends over time.

## Measurement Definitions

### Fitness Formula

In both training and evaluation, candidates play complete games with balanced starting positions (alternating first and second player). Draws receive half-credit:

$$\text{Fitness} = \frac{\text{wins} + 0.5 \times \text{draws}}{\text{total games}}$$

Because draws receive half-credit, a fitness score of 0.7 does not necessarily represent a 70% win rate. For example, a candidate that achieves 60 wins, 20 draws, and 20 losses over 100 games has a fitness score of:

$$\frac{60 + (0.5 \times 20)}{100} = 0.7000$$

Here, the fitness is 0.7000, but the actual win rate is 60%.

### Key Metrics Distinctions

- **Final-generation best:** The highest individual candidate fitness measured in the final evaluated generation of a training run against that run's fixed training opponent.
- **Final-generation mean:** The arithmetic average fitness across the entire population of candidates in that final evaluated generation.
- **Held-out evaluation:** Separate, post-training evaluation games where a candidate with frozen weights plays against opponents outside training (e.g., opponents initialized with seeds 1000–1002 or 2000–2002) that were never encountered during evolution.

## Selection

### Variable Changed
Parent selection mechanism: **Tournament selection** (tournament size $k=3$, sampling with replacement) versus **Truncation selection** (deterministic selection of the top $N$ candidates).

### Settings Held Constant
- Population size: 8
- Parent count: 2
- Elite count: 1 (top candidate preserved unchanged)
- Mutation noise: Fixed standard deviation $\sigma = 0.10$
- Crossover: Disabled
- Evaluated generations: 5 (generations 0 through 4)
- Games per candidate: 20 games per generation against the static, untrained opponent
- Initialization: Conditions use the same seed within each pair; their recorded generation-0 candidate outcomes match.

### Seeds and Game Budget
- Seeds: 6 paired runs (seeds 0 through 5)
- Game budget: 5 generations × 8 candidates × 20 games = 800 candidate games per run (12 runs total = 9,600 games).

### Results

| Seed | Method | Final best fitness | Final population mean | Source |
| :--- | :--- | ---: | ---: | :--- |
| 0 | Tournament | 0.7000 | 0.55625 | [run_2026_10_13T__21_37_12Z.json](experiments/selections/run_2026_10_13T__21_37_12Z.json) |
| 0 | Truncation | 0.6000 | 0.49375 | [run_2026_10_13T__21_37_27Z.json](experiments/selections/run_2026_10_13T__21_37_27Z.json) |
| 1 | Tournament | 0.7000 | 0.521875 | [run_2026_10_13T__21_37_36Z.json](experiments/selections/run_2026_10_13T__21_37_36Z.json) |
| 1 | Truncation | 0.7000 | 0.559375 | [run_2026_10_13T__21_37_47Z.json](experiments/selections/run_2026_10_13T__21_37_47Z.json) |
| 2 | Tournament | 0.6500 | 0.546875 | [run_2026_10_13T__21_37_53Z.json](experiments/selections/run_2026_10_13T__21_37_53Z.json) |
| 2 | Truncation | 0.6250 | 0.471875 | [run_2026_10_13T__21_37_58Z.json](experiments/selections/run_2026_10_13T__21_37_58Z.json) |
| 3 | Tournament | 0.7250 | 0.52500 | [run_2026_10_13T__21_38_04Z.json](experiments/selections/run_2026_10_13T__21_38_04Z.json) |
| 3 | Truncation | 0.7000 | 0.53750 | [run_2026_10_13T__21_38_09Z.json](experiments/selections/run_2026_10_13T__21_38_09Z.json) |
| 4 | Tournament | 0.5250 | 0.41875 | [run_2026_10_13T__21_38_16Z.json](experiments/selections/run_2026_10_13T__21_38_16Z.json) |
| 4 | Truncation | 0.5750 | 0.51250 | [run_2026_10_13T__21_38_21Z.json](experiments/selections/run_2026_10_13T__21_38_21Z.json) |
| 5 | Tournament | 0.7000 | 0.559375 | [run_2026_10_13T__21_38_27Z.json](experiments/selections/run_2026_10_13T__21_38_27Z.json) |
| 5 | Truncation | 0.6000 | 0.446875 | [run_2026_10_13T__21_38_40Z.json](experiments/selections/run_2026_10_13T__21_38_40Z.json) |

### Interpretation

**Demonstrated Findings:**
- Tournament selection produced a higher final-generation best fitness in 4 of the 6 seeds (seeds 0, 2, 3, and 5), tied in seed 1 (both at 0.7000), and Truncation selection produced a higher best fitness in seed 4 (0.5750 vs. 0.5250).
- For final population mean fitness, results split evenly: Tournament was higher in 3 seeds (seeds 0, 2, and 5), while Truncation was higher in 3 seeds (seeds 1, 3, and 4).
- Final-generation performance changed substantially across random seeds. In seed 4, neither method exceeded 0.5750 final-generation best fitness, whereas seeds 0, 1, 3, and 5 reached 0.7000 or higher. Earlier generations in seed 4 scored higher: 0.7250 for tournament and 0.7500 for truncation, so the final values are not the run-wide peaks.
- Best fitness and population mean fitness do not tell the same story. In seed 1, best fitness was identical, but truncation achieved a higher population mean (0.559375 vs. 0.521875). In seed 3, tournament produced the superior individual candidate (0.7250 vs. 0.7000), but truncation produced a higher population average (0.53750 vs. 0.52500).

**Possible Explanations (Hypotheses):**
- Tournament selection samples small sub-groups with replacement, which may give lower-ranked candidates an occasional chance to reproduce. This can maintain greater genetic diversity and help peak candidates discover better strategies. Truncation selection strictly restricts parents to the top two candidates, which may pull up the population mean more uniformly but risk premature convergence around local optima.
- However, with only 5 generations and 20 evaluation games per candidate, game outcome variance is significant. These small runs do not prove that tournament selection generally outperforms truncation selection.

## Annealing

### Variable Changed
Mutation noise schedule: **Fixed mutation strength** ($\sigma = 0.10$) versus **Annealed mutation strength** (decaying linearly across generations from $\sigma = 0.10$ down to $\sigma = 0.02$).

### Settings Held Constant
- Selection method: Truncation selection
- Crossover: Disabled
- Population size: 8
- Parent count: 2
- Elite count: 1
- Evaluated generations: 6 (generations 0 through 5)
- Games per candidate: 20 games per generation against the static, untrained opponent
- Initial population: Initial candidates and generation-0 scores match within each paired seed.

*File selection note:* The `experiments/annealing/` folder contains 11 JSON files, including historical crossover-enabled runs. Only the six matched records linked in the table below, where crossover was disabled, are used for the schedule comparison.

### Seeds and Game Budget
- Seeds: 3 paired runs (seeds 0, 1, and 2)
- Game budget: 6 generations × 8 candidates × 20 games = 960 candidate games per run (6 runs total = 5,760 games).

### Results

| Seed | Schedule | Final best fitness | Final population mean | Source |
| :--- | :--- | ---: | ---: | :--- |
| 0 | Fixed (0.10) | 0.6000 | 0.500000 | [run_2026_10M_26D__22h_59m_32s.json](experiments/annealing/run_2026_10M_26D__22h_59m_32s.json) |
| 0 | Annealed (0.10 → 0.02) | 0.5500 | 0.471875 | [run_2026_10M_26D__23h_21m_13s.json](experiments/annealing/run_2026_10M_26D__23h_21m_13s.json) |
| 1 | Fixed (0.10) | 0.7000 | 0.596875 | [run_2026_10M_26D__22h_59m_49s.json](experiments/annealing/run_2026_10M_26D__22h_59m_49s.json) |
| 1 | Annealed (0.10 → 0.02) | 0.7500 | 0.584375 | [run_2026_10M_26D__23h_21m_32s.json](experiments/annealing/run_2026_10M_26D__23h_21m_32s.json) |
| 2 | Fixed (0.10) | 0.6250 | 0.540625 | [run_2026_10M_26D__23h_00m_07s.json](experiments/annealing/run_2026_10M_26D__23h_00m_07s.json) |
| 2 | Annealed (0.10 → 0.02) | 0.6000 | 0.465625 | [run_2026_10M_26D__23h_21m_49s.json](experiments/annealing/run_2026_10M_26D__23h_21m_49s.json) |

### Interpretation

**Demonstrated Findings:**
- Fixed mutation produced a higher final best fitness in 2 of the 3 seeds (seed 0: 0.6000 vs. 0.5500; seed 2: 0.6250 vs. 0.6000), while Annealed mutation produced a higher best fitness in seed 1 (0.7500 vs. 0.7000).
- For final population mean fitness, Fixed mutation produced higher scores across all 3 seeds (seed 0: 0.500000 vs. 0.471875; seed 1: 0.596875 vs. 0.584375; seed 2: 0.540625 vs. 0.465625).
- Performance outcomes varied across seeds. While fixed mutation led in seeds 0 and 2, the single highest candidate fitness across all annealing runs was achieved under annealed mutation in seed 1 (0.7500).
- Best and mean fitness told the same story in seeds 0 and 2 (fixed mutation led on both metrics), but disagreed in seed 1: annealed mutation found a higher peak individual score (0.7500 vs. 0.7000), while fixed mutation maintained a higher average across the population (0.596875 vs. 0.584375).

**Possible Explanations (Hypotheses):**
- In theory, annealing allows broader exploration early in training and finer optimization later. However, over a short run of only 6 generations, reducing the mutation scale down to 0.02 so rapidly may restrict exploration too early, making it difficult for offspring to escape suboptimal plateaus unless a strong candidate appeared in the first two generations.
- Because each condition was tested across only 3 seeds with 20 evaluation games per candidate, these data points reflect small-sample exploratory runs rather than general proof of one schedule's superiority.

## Crossover

### Variable Changed
Genetic operator: **Uniform crossover enabled** (coordinate-wise probability of 0.5 of inheriting weights from parent B, followed by mutation) versus **Crossover disabled** (mutation only). Both conditions are evaluated across two separate selection regimes: Truncation selection and Tournament selection.

### Settings Held Constant
- Mutation schedule: Annealing enabled ($\sigma = 0.10 \to 0.02$)
- Population size: 8
- Parent count: 2
- Elite count: 1
- Evaluated generations: 6 (generations 0 through 5)
- Games per candidate: 20 games per generation against the static, untrained opponent
- Tournament size: $k=3$ for tournament runs

*Shared record note:* The three Truncation / Crossover Off records are the exact same historical runs used as the Annealed condition in the annealing comparison above. They are linked to `experiments/annealing/` to reflect that they represent one single set of runs rather than independent experiments.

### Seeds and Game Budget
- Seeds: 6 matched pairs across 3 seeds (seeds 0, 1, and 2) evaluated under both Truncation and Tournament selection (12 runs evaluated).
- Game budget: 6 generations × 8 candidates × 20 games = 960 candidate games per run (12 runs = 11,520 games).

### Results

| Selection mode | Seed | Condition | Final best fitness | Final population mean | Source |
| :--- | :--- | :--- | ---: | ---: | :--- |
| Truncation | 0 | Crossover On | 0.7250 | 0.51875 | [run_2026_10M_26D__23h_21m_01s.json](experiments/crossover/run_2026_10M_26D__23h_21m_01s.json) |
| Truncation | 0 | Crossover Off | 0.5500 | 0.471875 | [run_2026_10M_26D__23h_21m_13s.json](experiments/annealing/run_2026_10M_26D__23h_21m_13s.json) |
| Truncation | 1 | Crossover On | 0.6250 | 0.500000 | [run_2026_10M_26D__23h_21m_23s.json](experiments/crossover/run_2026_10M_26D__23h_21m_23s.json) |
| Truncation | 1 | Crossover Off | 0.7500 | 0.584375 | [run_2026_10M_26D__23h_21m_32s.json](experiments/annealing/run_2026_10M_26D__23h_21m_32s.json) |
| Truncation | 2 | Crossover On | 0.5500 | 0.434375 | [run_2026_10M_26D__23h_21m_41s.json](experiments/crossover/run_2026_10M_26D__23h_21m_41s.json) |
| Truncation | 2 | Crossover Off | 0.6000 | 0.465625 | [run_2026_10M_26D__23h_21m_49s.json](experiments/annealing/run_2026_10M_26D__23h_21m_49s.json) |
| Tournament | 0 | Crossover On | 0.6000 | 0.475000 | [run_2026_10M_26D__23h_25m_44s.json](experiments/crossover/run_2026_10M_26D__23h_25m_44s.json) |
| Tournament | 0 | Crossover Off | 0.7000 | 0.528125 | [run_2026_10M_26D__23h_25m_50s.json](experiments/crossover/run_2026_10M_26D__23h_25m_50s.json) |
| Tournament | 1 | Crossover On | 0.6500 | 0.496875 | [run_2026_10M_26D__23h_25m_56s.json](experiments/crossover/run_2026_10M_26D__23h_25m_56s.json) |
| Tournament | 1 | Crossover Off | 0.5750 | 0.481250 | [run_2026_10M_26D__23h_26m_02s.json](experiments/crossover/run_2026_10M_26D__23h_26m_02s.json) |
| Tournament | 2 | Crossover On | 0.7750 | 0.521875 | [run_2026_10M_26D__23h_26m_07s.json](experiments/crossover/run_2026_10M_26D__23h_26m_07s.json) |
| Tournament | 2 | Crossover Off | 0.5500 | 0.487500 | [run_2026_10M_26D__23h_26m_21s.json](experiments/crossover/run_2026_10M_26D__23h_26m_21s.json) |

### Interpretation

**Demonstrated Findings:**
- Under Truncation selection, Crossover On won in 1 of 3 seeds (seed 0: 0.7250 vs. 0.5500 best, 0.51875 vs. 0.471875 mean), while Crossover Off won in seeds 1 and 2.
- Under Tournament selection, Crossover On won in 2 of 3 seeds (seed 1: 0.6500 vs. 0.5750 best, 0.496875 vs. 0.481250 mean; seed 2: 0.7750 vs. 0.5500 best, 0.521875 vs. 0.487500 mean), while Crossover Off won in seed 0 (0.7000 vs. 0.6000 best, 0.528125 vs. 0.475000 mean).
- Performance changed across seeds in both selection modes; the winning condition was not consistent across random seeds.
- Best fitness and population mean fitness moved in the exact same direction for every single seed in both selection regimes. Whenever a condition had a higher final best fitness, it also had a higher final population mean.

**Possible Explanations (Hypotheses):**
- In neural network neuroevolution, crossing over individual weights coordinate-by-coordinate often breaks apart co-adapted hidden representations (the permutation problem or competing conventions), which may explain why crossover did not reliably beat mutation alone under truncation.
- Under tournament selection, maintaining slightly greater candidate diversity might occasionally yield complementary parents, which could explain the high individual fitness in Tournament seed 2 (0.7750).
- Additionally, parent selection allows the same candidate to be chosen twice, meaning some crossover events simply combine two identical networks. Given only 3 seeds per comparison, these runs do not demonstrate that crossover consistently improves or harms performance over mutation alone.
