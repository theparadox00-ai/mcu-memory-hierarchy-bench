# _template_board

Copy this folder to start a new board:

```
cp -r _template_board <board_name>
```

```
<board_name>/
  README.md      clocks, memory map, memory attributes, compiler flags, how to flash
  port/          port.c (+ port_overheads.c), Makefile, startup, linker script, vendor headers
  placements/    nn_place.c/.h per model and SRAM budget for this board
  experiments/   one folder per experiment: Makefile, README.md, results/
```

Steps are in [docs/PORTING.md](../docs/PORTING.md).
