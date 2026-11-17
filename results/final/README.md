# Final results (9x9 only)

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

## E3 · Famous hard puzzles: solved starts / starts

DR: 100 random starts × 10,000 iterations; PG / APG / FW: 20 starts × 3000 iterations. Paper (Table 3, 1000 starts): AI Escargot 985, 'Nasty' 202.

| Puzzle | pg | apg | fw | dr |
| --- | --- | --- | --- | --- |
| AI Escargot | 1/20 | 1/20 | 0/20 | 100/100 |
| Nasty (paper Fig. 11) | 0/20 | 0/20 | 0/20 | 7/100 |
| Inkala 2012 | 0/20 | 0/20 | 1/20 | 100/100 |
| Easter Monster | 0/20 | 0/20 | 0/20 | 87/100 |
