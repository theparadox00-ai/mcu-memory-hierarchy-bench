# host

Runs the engine and the `placement_sweep` study on a PC. External memory is emulated by a static array, so timings are not meaningful; the purpose is to check that a model and its placements are correct before flashing a board.

```
cd host/port
make MODEL=<model> SRAM_KB=<KB> BOARD=<board> run
```

`BOARD` selects whose placements to use (default `stm32h757i_eval`). The output must say `all layers match`, and the checksum must equal the one printed on the board.
