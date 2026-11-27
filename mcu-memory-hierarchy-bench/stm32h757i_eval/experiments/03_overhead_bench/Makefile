PORT_DIR := ../../port
PORT  := stm32h757i_eval
BUILD := $(abspath build)

all:
	$(MAKE) --no-print-directory -C $(PORT_DIR) STUDY=overhead_bench BUILD=$(BUILD)

clean:
	rm -rf build

.PHONY: all clean
