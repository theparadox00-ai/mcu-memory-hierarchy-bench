# stm32h757i_eval

Everything for this board lives in this folder:

```
stm32h757i_eval/
  port/          board code (clocks, UART, FMC SDRAM, MPU/caches, MDMA), Makefile, startup, linker, CMSIS
  placements/    nn_place.c/.h per model and SRAM budget, generated for this board
  experiments/   01_placement_sweep_cached, 02_placement_sweep_noncached, 03_overhead_bench
```

## Setup
| Item | Value |
|---|---|
| Board | STM32H757I-EVAL, Cortex-M7 core (CM4 idle) |
| Power | Direct SMPS, VOS1 |
| Clock | HSE 25 MHz → PLL1 (M 5, N 160, P 2) → 400 MHz; AXI/AHB 200 MHz; APB 100 MHz; flash 2 WS |
| Fast tier | AXI SRAM @ 0x24000000 (activation arena) |
| Slow tier | IS42S32800J SDRAM, 32 MB, FMC bank 2 @ 0xD0000000, 32-bit, SDCLK 100 MHz |
| Scratch | DTCM @ 0x20000000 |
| Weights | internal flash |
| Timing | DWT cycle counter, ms = cycles / 400000 |
| Console | USART1 PB14/PB15 → ST-LINK VCP, 115200 8N1 |

SDRAM: 9 column / 12 row bits, 4 banks, CAS 3; TMRD 2, TXSR 7, TRAS 4, TRC 7, TWR 2, TRP 2, TRCD 2; refresh 0x603; mode 0x230.

### MPU (once at boot)
| Region | Base | Size | Attributes |
|---|---|---|---|
| 0 | 0xD0000000 | 32 MB | Normal, write-back, read/write-allocate (TEX=1 C=1 B=1), non-shareable, XN, full access |

PRIVDEFENA = 1; I-cache and D-cache on.

### Compiler
```
arm-none-eabi-gcc -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -std=gnu11 -O3 -g3
  -ffunction-sections -fdata-sections -Wall -DCORE_CM7 -DSTM32H757xx --specs=nano.specs
link: --specs=nosys.specs --specs=nano.specs -Wl,--gc-sections -static -lc -lm
```

## Memory option
`EXTMEM_CACHEABLE=1` (default): region 0 is Normal write-back read/write-allocate.
`EXTMEM_CACHEABLE=0`: region 0 is Normal non-cacheable (TEX=1 C=0 B=0). Everything else is identical.

## Build
```
make STUDY=placement_sweep MODEL=<model> SRAM_KB=<512|256|128|0> [EXTMEM_CACHEABLE=0|1] [BUILD=<dir>]
make STUDY=overhead_bench [BUILD=<dir>]
```
(run from `port/`). The experiment folders under `experiments/` call this Makefile with the right options.

## Notes
- Kernels are plain C (SMLAD for 1×1 convolutions), not an optimised library.
- Measured results in `experiments/` were taken with the per-configuration STM32CubeIDE projects that preceded this layout; this port uses the same kernels, harness logic, clock, memory and MPU settings.
