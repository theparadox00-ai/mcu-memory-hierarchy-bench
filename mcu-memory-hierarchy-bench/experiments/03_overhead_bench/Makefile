ROOT  := ../..
PORT  ?= stm32h757i_eval
BUILD := $(abspath build/$(PORT))

all:
	$(MAKE) --no-print-directory -C $(ROOT)/ports/$(PORT) STUDY=overhead_bench BUILD=$(BUILD)

clean:
	rm -rf build

.PHONY: all clean
