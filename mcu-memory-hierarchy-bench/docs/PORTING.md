# Adding a new MCU

1. **Copy the template**
   ```
   cp -r ports/_template ports/<board>
   ```
2. **Implement `port.c`** (see the contract in `ARCHITECTURE.md`):
   - clock tree to the frequency you want to report, and `port_cpu_hz()` to match;
   - a cycle counter for `port_cycles()` (DWT on Cortex-M3/4/7/33/55/85; a timer otherwise);
   - a UART for `port_putc()`;
   - external / slow memory bring-up in `port_extmem_init()` (SDRAM, PSRAM, HyperRAM, OSPI RAM…), and its address window;
   - memory attributes in `port_memory_config()` (MPU regions, caches) — configured once;
   - a fast scratch area of at least 32 KB for `port_scratch()` (TCM if available).
3. **Implement `port_overheads.c`** if the board should run `overhead_bench` (MPU, cache, DMA). Leave it out otherwise.
4. **Add vendor files**: startup, linker script, device headers (`cmsis/`).
5. **Write a Makefile** — copy `ports/stm32h757i_eval/Makefile` and change `ARCH`, defines, startup and linker paths.
6. **Generate placements for the board's memory budgets**
   ```
   python3 tools/make_placement.py <model>.tflite <plan> placements/<board>/<model>/s<KB> models/<model>/nn_model.c
   ```
7. **Validate**: the first line after boot must read `verify vs TFLite: all layers match`, and the output checksum must equal the one from `ports/host`.
8. **Document the board** in `ports/<board>/README.md` (clocks, memory map, attributes, compiler flags) and run the experiments with `make PORT=<board>`; store results in each experiment's `results/` folder with the board name in the CSV.

## Checklist for comparable numbers
- Same model files and placements as other boards (only the port changes).
- Report the real core clock; never assume a nominal one.
- Debugger disconnected while timing.
- Warm-up runs before timed runs; store timings in RAM, print after the loop.
- Document memory attributes for every tier.
