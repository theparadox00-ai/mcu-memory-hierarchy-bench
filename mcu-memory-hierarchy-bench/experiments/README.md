# Experiments

Each folder is one experiment: a `Makefile` that builds it for a port, a `README.md` with the question, configuration and results, and a `results/` folder with the CSV and plots.

| # | Folder | Study | What changes |
|---|---|---|---|
| 01 | `01_placement_sweep_cached` | placement_sweep | SRAM budget 512/256/128/0 KB; external memory cacheable |
| 02 | `02_placement_sweep_noncached` | placement_sweep | same sweep; external memory non-cacheable |
| 03 | `03_overhead_bench` | overhead_bench | MPU reprogram, D-cache clean+invalidate, DMA fast→slow copy |

Build any experiment for another board with `make PORT=<board>`.

To add an experiment: create `NN_<name>/` with a `Makefile` that calls `ports/$(PORT)/Makefile` with its options, a `README.md`, and a `results/` folder.
