PORT_DIR := ../../port
PORT     := my_board
MODELS   ?= mobilenetv2_a035_224_fft_int8 miniresnetv2_s2_64x50_tl_int8
BUDGETS  ?= 512 256 128 0
BUILD    := $(abspath build)

all:
	@for m in $(MODELS); do for s in $(BUDGETS); do \
	  $(MAKE) --no-print-directory -C $(PORT_DIR) STUDY=placement_sweep MODEL=$$m SRAM_KB=$$s \
	    BUILD=$(BUILD) LABEL=$(PORT)_$${m}_s$${s} || exit 1; \
	done; done

clean:
	rm -rf build

.PHONY: all clean
