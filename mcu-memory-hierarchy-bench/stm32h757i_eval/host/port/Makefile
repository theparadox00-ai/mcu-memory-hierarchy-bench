ROOT    := ../..
STUDY   ?= placement_sweep
MODEL   ?= mobilenetv2_a035_224_fft_int8
SRAM_KB ?= 512
BOARD   ?= stm32h757i_eval
N_RUNS  ?= 5
CC      ?= gcc

PLACE  := $(ROOT)/$(BOARD)/placements/$(MODEL)/s$(SRAM_KB)
OUT    := build/$(MODEL)/s$(SRAM_KB)
CFLAGS := -std=gnu11 -O2 -Wall -Wextra -I$(ROOT)/core/include -I$(ROOT)/core/engine \
          -I$(ROOT)/models/$(MODEL) -I$(PLACE) -DN_RUNS=$(N_RUNS)u -DN_WARMUP=1u \
          -DEXTMEM_TEST_BYTES=1048576u -DBENCH_LABEL=\"host_$(MODEL)_s$(SRAM_KB)\"
SRCS   := port.c $(ROOT)/core/bench/common.c $(ROOT)/core/bench/placement_sweep.c $(ROOT)/core/engine/nn_engine.c \
          $(ROOT)/models/$(MODEL)/nn_model.c $(ROOT)/models/$(MODEL)/nn_weights.c $(PLACE)/nn_place.c

all: $(OUT)/bench

$(OUT)/bench: $(SRCS) Makefile
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(SRCS) -lm -o $@

run: all
	./$(OUT)/bench

clean:
	rm -rf build

.PHONY: all run clean
