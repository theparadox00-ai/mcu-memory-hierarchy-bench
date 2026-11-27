# 03 — Memory-management overheads

**Question:** what do the primitives for managing a fast/slow memory hierarchy cost?

| Item | Value |
|---|---|
| Study code | `core/bench/overhead_bench.c` |
| Board code | `stm32h757i_eval/port/port_overheads.c` |
| Iterations | 10 warm-up + 1000 timed per test, DWT cycles |
| Test buffer | 128 KB in AXI SRAM (128 KB aligned) |
| External memory | SDRAM, MPU region 0 Normal write-back read/write-allocate, never changed |

## Tests
1. **MPU region reprogram** — `MPU->CTRL = 0` → configure region 1 (the AXI SRAM buffer) → `MPU->CTRL = PRIVDEFENA | ENABLE` → `__DSB(); __ISB();`.
2. **D-cache clean + invalidate** — buffer fully rewritten (not timed), then `SCB_CleanInvalidateDCache_by_Addr(buf, 131072)` timed.
3. **DMA copy fast → slow** — MDMA channel 0, software request, 64-bit beats, 16-beat bursts, 4 × 32 KB blocks, AXI SRAM → SDRAM. Source cleaned before the timer; timed from request to transfer-complete; copy verified.

## Build
```
make
```
ELF: `build/overhead_bench/bench.elf`

## Results (`results/results.csv`)
| test | mean µs | max µs | min µs | mean cycles | MB/s |
|---|---|---|---|---|---|
| MPU region reprogram | 0.076 | 0.105 | 0.072 | 30 | — |
| D-cache clean + invalidate, 128 KB | 80.685 | 80.717 | 71.727 | 32 274 | — |
| MDMA copy 128 KB AXI SRAM → SDRAM | 333.582 | 333.640 | 333.455 | 133 432 | 392.9 |

DMA transfer errors: 0; copy check: OK.

## Observations
- An MPU reprogram is ~30 cycles — negligible next to an inference.
- Cache maintenance costs ~7.9 cycles per 32-byte line; the call walks all 4096 lines even though the D-cache is 16 KB.
- MDMA reaches ~98 % of the SDRAM bus peak (32 bit × 100 MHz = 400 MB/s); moving a 128 KB tensor costs 0.33 ms.
