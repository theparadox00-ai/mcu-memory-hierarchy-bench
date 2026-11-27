# Studies

## placement_sweep
**Question:** how does latency change as the fast-memory budget for activations shrinks and tensors move to the slow tier?

- Inputs: one model, one placement (`s<KB>`). Only the placement differs between configurations of a model.
- Each tensor lives wholly in one tier and is computed in place (no staging copies).
- Boot: external memory init + test → memory config → engine init → on-target verification against TFLite.
- Input: fixed LCG pattern (seed 12345), identical every run.
- `N_WARMUP` (default 10) warm-up runs, then `N_RUNS` (default 500) timed runs; cycle counts stored in RAM and printed after the loop.
- Output: per-run cycles, min/mean/max ms, output checksum, and one line
  `CSV,config,board,model,sram_budget_kb,act_sram_kb,act_ext_kb,min_ms,mean_ms,max_ms,out_checksum`.

Build knobs: `MODEL`, `SRAM_KB`, `LABEL`, `-DN_RUNS=…`, `-DN_WARMUP=…`, port-specific memory options (e.g. `EXTMEM_CACHEABLE` on `stm32h757i_eval`).

## overhead_bench
**Question:** what do the primitives for managing a memory hierarchy cost?

- MPU region reprogram (disable → configure → enable → barriers).
- D-cache clean + invalidate over a dirtied fast-memory buffer.
- DMA copy fast → slow tier, with throughput; the copy is verified.
- `N_WARMUP` (10) + `N_ITER` (1000) iterations each; output mean/max/min µs, mean cycles, MB/s:
  `CSV,board,test,mean_us,max_us,min_us,mean_cycles,MB_per_s`.

## Adding a study
1. Add `core/bench/<study>.c` with a `main()` that uses only `port.h` (and optional headers).
2. If it needs new board capabilities, declare them in a new `core/include/port_<feature>.h`; ports opt in by implementing it.
3. Add a `STUDY=<study>` branch to each port Makefile.
4. Add an experiment folder under `experiments/` that builds it.
5. Document it here.
