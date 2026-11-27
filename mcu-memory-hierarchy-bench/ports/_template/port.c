#include "port.h"

void port_init(void)
{
}

const char *port_name(void) { return "my_board"; }

uint32_t port_cpu_hz(void) { return 0u; }

uint32_t port_cycles(void)
{
    return 0u;
}

void port_putc(char c)
{
    (void)c;
}

void port_flush(void)
{
}

int port_extmem_init(void)
{
    return 0;
}

uintptr_t port_extmem_base(void) { return 0u; }

uint32_t port_extmem_bytes(void) { return 0u; }

void port_memory_config(void)
{
}

const char *port_memory_desc(void) { return ""; }

void *port_scratch(uint32_t *bytes)
{
    *bytes = 0u;
    return 0;
}

void port_halt(void)
{
    while (1) {}
}
