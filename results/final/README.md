# Final results

Produced by `bash experiments/run_final.sh` (Nov 15, 2026). Same budget for every method: at most 3000 iterations per start. 9×9 puzzles: `data/dev_sample.csv` (300 Kaggle puzzles, 50 per difficulty band). Figures: `results/figures/main/` and `results/figures/restarts/`.

## E1 · 9×9, one start (share solved)

| band | pg | apg | fw | dr |
| --- | --- | --- | --- | --- |
| 0 | 100 % | 100 % | 100 % | 100 % |
| 0.1-1.9 | 78 % | 78 % | 58 % | 100 % |
| 2-2.9 | 52 % | 52 % | 48 % | 100 % |
| 3-3.9 | 38 % | 38 % | 32 % | 98 % |
| 4-4.9 | 34 % | 32 % | 34 % | 96 % |
| 5-8.5 | 34 % | 32 % | 18 % | 96 % |
| **total** | **168/300** | **166/300** | **145/300** | **295/300** |

## E2 · 9×9, up to 5 starts × 1000 iterations

| band | pg | apg | fw | dr |
| --- | --- | --- | --- | --- |
| 0 | 100 % | 100 % | 100 % | 100 % |
| 0.1-1.9 | 88 % | 88 % | 94 % | 100 % |
| 2-2.9 | 66 % | 70 % | 80 % | 100 % |
| 3-3.9 | 50 % | 50 % | 54 % | 100 % |
| 4-4.9 | 46 % | 46 % | 50 % | 100 % |
| 5-8.5 | 38 % | 38 % | 50 % | 100 % |
| **total** | **194/300** | **196/300** | **214/300** | **300/300** |

## E3 · 16×16 generated (share solved)

| band | pg | apg | fw | dr |
| --- | --- | --- | --- | --- |
| 40% empty | 80 % | 100 % | 45 % | 100 % |
| 50% empty | 95 % | 95 % | 0 % | 100 % |
| 60% empty | 0 % | 10 % | 10 % | 95 % |
| **total** | **35/60** | **41/60** | **11/60** | **59/60** |

## E3 · 25×25 generated (share solved)

| band | pg | apg | fw | dr |
| --- | --- | --- | --- | --- |
| 40% empty | 100 % | 100 % | 40 % | 100 % |
| 50% empty | 0 % | 0 % | 5 % | 100 % |
| 60% empty | 0 % | 0 % | 0 % | 85 % |
| **total** | **20/60** | **20/60** | **9/60** | **57/60** |

## E3 · Cost of one iteration (median, ms)

| N | pg | apg | fw | dr |
| --- | --- | --- | --- | --- |
| 9×9 | 0.21 | 0.29 | 0.35 | 0.13 |
| 16×16 | 0.44 | 0.59 | 0.75 | 0.31 |
| 25×25 | 1.07 | 1.42 | 1.73 | 0.83 |

## E4 · Famous hard puzzles: solved starts / starts

DR: 100 random starts × 10,000 iterations; PG / APG / FW: 20 starts × 3000 iterations. Paper (Table 3, 1000 starts): AI Escargot 985, 'Nasty' 202.

| Puzzle | pg | apg | fw | dr |
| --- | --- | --- | --- | --- |
| AI Escargot | 1/20 | 1/20 | 0/20 | 100/100 |
| Nasty (paper Fig. 11) | 0/20 | 0/20 | 0/20 | 7/100 |
| Inkala 2012 | 0/20 | 0/20 | 1/20 | 100/100 |
| Easter Monster | 0/20 | 0/20 | 0/20 | 87/100 |
