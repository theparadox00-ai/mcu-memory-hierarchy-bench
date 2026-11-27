# Roadmap

## Targets
- More MCUs with different fast/slow memory ratios (other Cortex-M7/M55/M85 parts, RISC-V MCUs, ESP32-S3 with PSRAM).
- Additional slow tiers: OctoSPI/HyperRAM, QSPI PSRAM, SD card.

## Memory system
- Explicit staging: DMA hot tensors into fast memory ahead of use (double buffering) instead of computing in place.
- Row/tile streaming placements (partial tensors in fast memory) versus whole-tensor placement.
- Weights in slow memory (external flash / RAM) in addition to activations.
- TCM as a separate tier.

## Compute
- Alternative kernel backends (e.g. CMSIS-NN) to see how placement sensitivity changes with faster compute.
- Per-layer timing to attribute latency to individual tensors and tiers.

## Measurement
- Energy per inference alongside latency.
- Automatic log capture and CSV/plot generation from the serial stream.
- CI build of every port × study × model × placement.
