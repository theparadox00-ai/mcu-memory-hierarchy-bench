# mcu-memory-hierarchy-bench

**A portable, bare-metal framework for measuring how the placement of neural-network activations across a microcontroller's memory hierarchy affects inference latency.**

## Abstract

Microcontrollers have only a few hundred kilobytes of fast on-chip SRAM, while modern neural networks need far more memory for their intermediate data (activations) during inference. When the activations do not fit, they must be placed in slower memory such as external SDRAM, PSRAM or an SD card, and every such placement changes how fast the network runs.

This project measures that effect directly on real hardware. An int8 TensorFlow Lite model is run by a bare-metal inference engine that is bit-exact with TFLite. A placement plan decides, tensor by tensor, whether each activation lives in on-chip SRAM or in an external memory tier, and the model is then executed in place under a range of SRAM budgets. Every layer is verified against TFLite before timing, and latency is measured with the CPU cycle counter over hundreds of runs, so that only the memory placement changes between configurations.

The study asks how much latency grows as activations move off-chip, how much of that cost the data cache hides, what it costs to manage the hierarchy explicitly (MPU reconfiguration, cache maintenance, DMA transfers), and whether these conclusions hold across different microcontrollers. The framework keeps all board-specific code in a small per-board port, so new MCUs, models and experiments can be added without changing the rest.

Full documentation, experiments and results: [mcu-memory-hierarchy-bench/README.md](mcu-memory-hierarchy-bench/README.md)
