#ifndef PORT_H
#define PORT_H

#include <stdint.h>

void port_init(void);
const char *port_name(void);
uint32_t port_cpu_hz(void);
uint32_t port_cycles(void);
void port_putc(char c);
void port_flush(void);

int port_extmem_init(void);
uintptr_t port_extmem_base(void);
uint32_t port_extmem_bytes(void);

void port_memory_config(void);
const char *port_memory_desc(void);

void *port_scratch(uint32_t *bytes);

void port_halt(void) __attribute__((noreturn));

#endif
