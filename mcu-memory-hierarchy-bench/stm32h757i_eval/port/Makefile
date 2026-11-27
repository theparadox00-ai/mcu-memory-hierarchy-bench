ROOT     := ../..
BOARD    := stm32h757i_eval
STUDY    ?= placement_sweep
MODEL    ?= mobilenetv2_a035_224_fft_int8
SRAM_KB  ?= 512
EXTMEM_CACHEABLE ?= 1
PREFIX   ?= arm-none-eabi-
BUILD    ?= build

CC      := $(PREFIX)gcc
OBJCOPY := $(PREFIX)objcopy
SIZE    := $(PREFIX)size

ARCH    := -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb
CFLAGS  := $(ARCH) -std=gnu11 -O3 -g3 -ffunction-sections -fdata-sections -Wall -Wextra \
           -DCORE_CM7 -DSTM32H757xx -DEXTMEM_CACHEABLE=$(EXTMEM_CACHEABLE) \
           -I$(ROOT)/core/include -Icmsis
LDFLAGS := $(ARCH) -T linker/STM32H757XIHX_FLASH.ld --specs=nano.specs --specs=nosys.specs \
           -Wl,--gc-sections -static -Wl,--start-group -lc -lm -Wl,--end-group

PORT_SRCS := port.c syscalls.c sysmem.c startup/startup_stm32h757xihx.s $(ROOT)/core/bench/common.c

ifeq ($(STUDY),placement_sweep)
LABEL   ?= $(BOARD)_$(MODEL)_s$(SRAM_KB)
OUT     := $(BUILD)/$(STUDY)/$(MODEL)/s$(SRAM_KB)
PLACE   := ../placements/$(MODEL)/s$(SRAM_KB)
SRCS    := $(PORT_SRCS) $(ROOT)/core/bench/placement_sweep.c $(ROOT)/core/engine/nn_engine.c \
           $(ROOT)/models/$(MODEL)/nn_model.c $(ROOT)/models/$(MODEL)/nn_weights.c $(PLACE)/nn_place.c
CFLAGS  += -I$(ROOT)/core/engine -I$(ROOT)/models/$(MODEL) -I$(PLACE) -DBENCH_LABEL=\"$(LABEL)\"
else ifeq ($(STUDY),overhead_bench)
OUT     := $(BUILD)/$(STUDY)
SRCS    := $(PORT_SRCS) port_overheads.c $(ROOT)/core/bench/overhead_bench.c
else
$(error unknown STUDY $(STUDY))
endif

ELF := $(OUT)/bench.elf

all: $(ELF)

$(ELF): $(SRCS) Makefile
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -Wl,-Map=$(OUT)/bench.map -o $@
	$(OBJCOPY) -O binary $@ $(OUT)/bench.bin
	$(SIZE) $@

sweep:
	@for m in mobilenetv2_a035_224_fft_int8 miniresnetv2_s2_64x50_tl_int8; do \
	  for s in 512 256 128 0; do $(MAKE) --no-print-directory STUDY=placement_sweep MODEL=$$m SRAM_KB=$$s || exit 1; done; done
	@$(MAKE) --no-print-directory STUDY=overhead_bench

clean:
	rm -rf $(BUILD)

.PHONY: all sweep clean
