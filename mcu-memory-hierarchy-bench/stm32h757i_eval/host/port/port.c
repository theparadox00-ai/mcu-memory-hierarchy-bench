#include "port.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EXT_BYTES (4u * 1024u * 1024u)
#define SCRATCH_BYTES (128u * 1024u)

static int8_t ext_mem[EXT_BYTES] __attribute__((aligned(64)));
static int32_t scratch[SCRATCH_BYTES / 4u];

void port_init(void) {}
const char *port_name(void) { return "host"; }
uint32_t port_cpu_hz(void) { return 1000000000u; }

uint32_t port_cycles(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec);
}

void port_putc(char c) { putchar(c); }
void port_flush(void) { fflush(stdout); }
int port_extmem_init(void) { return 0; }
uintptr_t port_extmem_base(void) { return (uintptr_t)ext_mem; }
uint32_t port_extmem_bytes(void) { return EXT_BYTES; }
void port_memory_config(void) {}
const char *port_memory_desc(void) { return "host: external memory emulated by a static array"; }

void *port_scratch(uint32_t *bytes)
{
    *bytes = SCRATCH_BYTES;
    return scratch;
}

void port_halt(void)
{
    fflush(stdout);
    exit(0);
}
