ROOT     := ../..
PORT     ?= stm32h757i_eval
MODELS   ?= mobilenetv2_a035_224_fft_int8 miniresnetv2_s2_64x50_tl_int8
BUDGETS  ?= 512 256 128 0
EXTMEM_CACHEABLE := 1
BUILD    := $(abspath build/$(PORT))

all:
	@for m in $(MODELS); do for s in $(BUDGETS); do \
	  $(MAKE) --no-print-directory -C $(ROOT)/ports/$(PORT) STUDY=placement_sweep MODEL=$$m SRAM_KB=$$s \
	    EXTMEM_CACHEABLE=$(EXTMEM_CACHEABLE) BUILD=$(BUILD) LABEL=$(PORT)_$${m}_s$${s} || exit 1; \
	done; done

one:
	$(MAKE) --no-print-directory -C $(ROOT)/ports/$(PORT) STUDY=placement_sweep MODEL=$(MODEL) SRAM_KB=$(SRAM_KB) \
	  EXTMEM_CACHEABLE=$(EXTMEM_CACHEABLE) BUILD=$(BUILD) LABEL=$(PORT)_$(MODEL)_s$(SRAM_KB)

clean:
	rm -rf build

.PHONY: all one clean
