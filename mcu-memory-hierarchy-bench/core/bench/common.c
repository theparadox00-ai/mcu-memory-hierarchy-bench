#include "bench.h"
#include "port.h"
#include <stdio.h>

int __io_putchar(int ch)
{
    if (ch == '\n') port_putc('\r');
    port_putc((char)ch);
    return ch;
}

void bench_stats(const uint32_t *v, uint32_t n, bench_stats_t *s)
{
    s->min = 0xFFFFFFFFu;
    s->max = 0u;
    s->sum = 0u;
    s->n = n;
    for (uint32_t i = 0; i < n; i++) {
        s->sum += v[i];
        if (v[i] < s->min) s->min = v[i];
        if (v[i] > s->max) s->max = v[i];
    }
}

static void print_fixed3(uint64_t milli)
{
    printf("%lu.%03lu", (unsigned long)(milli / 1000u), (unsigned long)(milli % 1000u));
}

void bench_print_ms(uint64_t cycles_x, uint32_t div)
{
    uint64_t per_ms = port_cpu_hz() / 1000u;
    print_fixed3((cycles_x * 1000u) / (per_ms * div));
}

void bench_print_us(uint64_t cycles_x, uint32_t div)
{
    uint64_t per_us = port_cpu_hz() / 1000000u;
    print_fixed3((cycles_x * 1000u) / (per_us * div));
}

int bench_extmem_test(uintptr_t base, uint32_t bytes)
{
    volatile uint32_t *p = (volatile uint32_t *)base;
    uint32_t n = bytes / 4u;
    for (uint32_t i = 0; i < n; i++) p[i] = 0xA5A50000u ^ (i * 2654435761u);
    for (uint32_t i = 0; i < n; i++)
        if (p[i] != (0xA5A50000u ^ (i * 2654435761u))) return -(int)(i + 1u);
    return 0;
}
