# Final results (9x9 only)

Produced by `bash experiments/run_final.sh`, first on Nov 15, 2026 and re-run on the `9x9-only` branch on Nov 17, 2026 (36.7 minutes). The re-run reproduced every solved/unsolved outcome and iteration count exactly; only the `seconds` columns differ, and the CSVs here hold the Nov 17 timings. Same budget for every method within a start: at most 3000 iterations in E1, 1000 per start in E2. 9×9 puzzles: `data/dev_sample.csv` (300 Kaggle puzzles, 50 per difficulty band). Figures: `results/figures/main/`, `results/figures/restarts/` and `results/figures/analysis/`.

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

## Further metrics (E1, one start)

| | pg | apg | fw | dr |
| --- | --- | --- | --- | --- |
| Solved, 95 % CI (Wilson) | 50–62 % | 50–61 % | 43–54 % | 96–99 % |
| Correlation of success with difficulty | −0.48 | −0.49 | −0.50 | −0.13 |
| Median / mean iterations to solve | 250 / 262 | 220 / 232 | 257 / 357 | 493 / 567 |
| Median time per iteration (Nov 17 machine) | 0.27 ms | 0.36 ms | 0.50 ms | 0.16 ms |
| Median time to solve one puzzle | 0.066 s | 0.080 s | 0.109 s | 0.080 s |
| Total time, all 300 puzzles | 121 s | 161 s | 303 s | 30 s |
| Median constraint violation when unsolved | 1.00 | 1.00 | 1.00 | 1.05 |

Which methods solve each puzzle (E1): all four 124; DR plus one or two gradient methods 65; only DR 106; a gradient method but not DR 0; none 5 (DR's five misses, difficulty 3.1, 4.1, 4.8, 5.1 and 5.6). PG and APG have the same outcome on 296 of 300 puzzles.

## Findings

1. **Douglas–Rachford is far ahead and hardly depends on difficulty** (100 % → 96 % across bands, against 100 % → about 33 % for the course methods).
2. **The gradient methods fail by getting stuck, not by running out of iterations.** Unsolved runs end with a constraint violation near 1.0, a near-solution with a few conflicting entries. The `lam * x(1 - x)` term makes the objective nonconvex, and as `lam` grows the weights lock into the nearest 0/1 pattern. PG and APG finish every solve within about 550 iterations, so their 3000-iteration budget is never binding.
3. **Difficulty decides whether the gradient methods solve a puzzle, not how long they take**: PG's median iterations go only from 218 to 287 across bands, while DR's go from 353 to 619.
4. **Acceleration does not change the result.** APG needs about 12 % fewer iterations than PG but each costs about 34 % more (objective and gradient at both the extrapolated point and the current iterate), so it is slower in wall-clock time and reaches the same local minimum on 296 of 300 puzzles.
5. **Frank–Wolfe is weakest from one start** (lowest solve rate, costliest iteration because of its Armijo search from step 1, heavy tail of iterations) **but gains most from restarts** (+69 puzzles, from last to best of the gradient methods). Its outcome depends strongly on the start, so new starts reach different basins: it solved 42 puzzles on its 2nd start against 15 for PG.
6. **DR needs about twice the iterations but is the fastest overall**, because an iteration is only argmax projections and averages, with no gradient or line search.
7. **Restarts give diminishing returns.** Most solves come on start 1; PG ran all five starts on 116 puzzles and solved 10 of them on start 5. The hardest band stays at 38–50 % for the gradient methods.
8. **Hard puzzles:** the gradient methods solve 3 of 240 attempts. DR's AI Escargot result (100/100) is consistent with the paper's 985/1000, but its 'Nasty' result (7/100) is well below the paper's 202/1000: if our DR matched the paper, 7 or fewer out of 100 would happen with probability 0.0002. Every reading of the paper's text we tested gives 5–11 %; the remaining difference is most likely in the authors' unpublished C++ code (see "Why DR underperforms the paper on 'Nasty'" below).

## DR's five E1 misses with 10,000 iterations

One-off check (Nov 17, 2026): the five puzzles DR missed in E1, rerun from the same start (seed 0) with the paper's 10,000-iteration cap instead of 3000. The E2 column gives the iterations of the start that solved the puzzle (the CSV sums all starts; each failed start used 1000).

| Puzzle id | Difficulty | E1 (3000) | Same start, 10,000 | E2 (restarts) |
| --- | --- | --- | --- | --- |
| 2511252 | 5.1 | missed | solved at 4912 | start 2, 116 iterations |
| 2203850 | 5.6 | missed | solved at 3825 | start 4, 195 iterations |
| 495520 | 3.1 | missed | unsolved | start 2, 247 iterations |
| 1926592 | 4.1 | missed | unsolved | start 5, 123 iterations |
| 762699 | 4.8 | missed | unsolved | start 4, 327 iterations |

- With the paper's budget, DR's one-start result would be 297/300 instead of 295/300; both newly solved puzzles needed only a little more than 3000 iterations.
- The other three are stuck rather than slow: from this start DR wanders without converging even at 10,000 iterations, while a fresh start solves each in 123–327 iterations. For DR on these puzzles, restarting beats waiting.

Follow-up (Nov 17, 2026): 20 more random starts per puzzle, 10,000 iterations each, with a check for repeated states (all five copies equal to within 1e-9). All five puzzles have exactly one solution.

| Puzzle id | Difficulty | Other starts solved (of 20) | Median iterations when solved | Closest the E1 start gets |
| --- | --- | --- | --- | --- |
| 2511252 | 5.1 | 20 | 409 | solves at 4912 |
| 2203850 | 5.6 | 20 | 1158 | solves at 3825 |
| 495520 | 3.1 | 13 | 247 | 12 cells wrong |
| 1926592 | 4.1 | 8 | 533 | 25 cells wrong |
| 762699 | 4.8 | 7 | 561 | 11 cells wrong |

- 2511252 and 2203850 are easy for DR: every other start solves them; the E1 start was slow and hit the cap.
- 495520, 1926592 and 762699 are hard for DR: 35–65 % of starts solve them. None of the 63 runs on them revisits a state (unlike 'Nasty', where most runs loop); failing runs keep moving and never get closer than 11–25 wrong cells.
- What makes a puzzle hard for DR does not follow the Kaggle rating: the three DR-hard puzzles are rated 3.1–4.8, easier than the two DR finds easy (5.1 and 5.6). The data shows how DR fails on them, not which feature of the puzzles causes it.

## Why DR underperforms the paper on 'Nasty'

Investigation, Nov 17, 2026. All runs: paper order, start x0 = (y, …, y), 10,000 iterations, seeds 0–99 (the same starts as E3). Our run solves 7/100; the paper reports 202/1000 (20.2 %). Every number below is reproduced by `experiments/nasty_dr_variants.py` (`--tie raw|first|last|random`, `--givens partial|full|four`, `--puzzle "AI Escargot"`), about 2–3 minutes per configuration.

| Suspect | Test | Result |
| --- | --- | --- |
| Puzzle transcribed wrongly | Compared `data/hard_9.csv` with the paper's Figure 11 | Identical: every row's clues match, and the paper's solution satisfies all 22 clues |
| Algorithm differs from the paper | Checked sets C1–C5, the clue projection, T_{D,C} = (I + R_C R_D)/2, start, 10,000-iteration cap and stopping test against Sections 6.2–6.3 | All match the text |
| Start distribution | Not tested | Open. The paper says only "random y ∈ [0, 1]^(9×9×9)". The start keeps influencing the run: the clue projection passes every non-clue value through unchanged, so the clue copy carries real values from one iteration to the next. (An earlier version of this table called the start irrelevant, on the wrong assumption that all five copies are 0/1 after the first step.) |
| Tie-breaking in the one-hot projection (the paper leaves it open) | Lowest index / highest index / random among tied maxima, against `np.argmax` on raw floats | 5 / 9 / 11 / 7 solved. Ties occur in fewer than 0.1 % of group projections |
| Clue projection (the paper's C4 definition sets only the given entry to 1, its projection formula can be read as resetting the whole cell) | Given entry only / whole clue cell / 4-set version (clues fixed in the cell projection) | 7 / 9 / 5 solved. The whole-cell version still gives 99/100 on AI Escargot |
| Stopping test | Analysis | Ours (per-cell argmax, then a validity check) accepts every grid the paper's pointwise rounding accepts and more, so it can only raise our success rate |

Even the best variant (11/100) would occur only about 1 % of the time if the true rate were 20.2 %.

What the runs show about DR on this puzzle: with the current code, 77 of 100 runs revisit an earlier state (all five copies equal to within 1e-9), first at a median of iteration 5,644, and then repeat with a median period of 102 iterations; with the other deterministic tie rules it is 74 of 100 (periods of about 140–150). A deterministic run that revisits a state repeats from then on, so more iterations cannot help, which is why restarts beat longer runs. The paper itself reports 'Nasty' as unsolvable by another DR solver, 20 % with its own code and 24 % with the top-left clue removed, and attributes this to an abundance of near solutions; our implementation lands lower on the same hard puzzle.

Not tested: single vs double precision (ties are rare, so it is unlikely to matter much) and details of the authors' C++ code. Their companion page (carmamaths.org/DRmethods/comb-opt/) offers the source only on request by email, and its puzzle files are on a host that no longer resolves. Settling the gap requires that code.

## Caveats

- E1 and E2 do not have the same total budget: E1 allows at most 3000 iterations, E2 up to 5 × 1000 = 5000. For PG and APG the restart gain is still from new starts (their solves finish early); for FW and DR part of it may be extra budget. At 1000 iterations, DR's first start solves 265 puzzles against 295 at 3000.
- DR is capped at 3000 iterations in E1; the paper uses 10,000. With 10,000, two of its five misses are solved (see above).
- Timings depend on the machine: the Nov 17 run was 1.3–1.9× slower than Nov 15, but the ranking of methods did not change.
- With 50 puzzles per band, a 50 % share has a 95 % interval of about ±14 points; small band-to-band differences (e.g. PG 34 % vs APG 32 %) are noise.
- No hyperparameter tuning: results are for `lam_max = 0.6`, Armijo constant 0.25 and restart noise 0.3.
- The Kaggle difficulty rating measures difficulty for search-based solvers, not for a continuous relaxation.

## Figures in `results/figures/analysis/` (from `experiments/analysis_plots.py`)

| File | Shows |
| --- | --- |
| `overall_solve_rate.png` | solved out of 300 with 95 % intervals, one start vs. up to 5 starts |
| `restart_gain_by_band.png` | one panel per method: share solved per band, one start vs. restarts |
| `cost_per_method.png` | median time per iteration and total time for 300 puzzles |
| `iterations_to_solve.png` | box plot of iterations for solved puzzles (log scale) |
| `solved_overlap.png` | which methods solve each puzzle |
| `hard_puzzles.png` | share of random starts that solve each famous puzzle |
| `solved_on_start.png` | on which start each solved puzzle was solved |
