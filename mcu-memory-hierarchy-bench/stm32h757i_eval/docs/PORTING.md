# Adding a new MCU

1. **Copy the template**
   ```
   cp -r _template_board <board>
   ```
   Everything for the board stays inside `<board>/`: `port/`, `placements/`, `experiments/`, `README.md`.
2. **Implement `port/port.c`** (see the contract in `ARCHITECTURE.md`):
   - clock tree to the frequency you want to report, and `port_cpu_hz()` to match;
   - a cycle counter for `port_cycles()` (DWT on Cortex-M3/4/7/33/55/85; a timer otherwise);
   - a UART for `port_putc()`;
   - external / slow memory bring-up in `port_extmem_init()` (SDRAM, PSRAM, HyperRAM, OSPI RAM…), and its address window;
   - memory attributes in `port_memory_config()` (MPU regions, caches) — configured once;
   - a fast scratch area of at least 32 KB for `port_scratch()` (TCM if available).
3. **Implement `port/port_overheads.c`** if the board should run `overhead_bench` (MPU, cache, DMA). Leave it out otherwise.
4. **Add vendor files** to `port/`: startup, linker script, device headers.
5. **Write `port/Makefile`** — start from `port/Makefile.example` (the stm32h757i_eval one) and change `ARCH`, defines, startup and linker paths.
6. **Generate placements for the board's memory budgets**
   ```
   python3 tools/make_placement.py <model>.tflite <plan> <board>/placements/<model>/s<KB> models/<model>/nn_model.c
   ```
7. **Validate**: the board must print `verify vs TFLite: all layers match`, and the output checksum must equal the one from `host/`.
8. **Create experiments** under `<board>/experiments/NN_<name>/` (copy from `stm32h757i_eval/experiments/`, change `PORT`), document the board in `<board>/README.md`, and store results in each experiment's `results/` folder.

## Checklist for comparable numbers
- Same model files and placements as other boards (only the port changes).
- Report the real core clock; never assume a nominal one.
- Debugger disconnected while timing.
- Warm-up runs before timed runs; store timings in RAM, print after the loop.
- Document memory attributes for every tier.
