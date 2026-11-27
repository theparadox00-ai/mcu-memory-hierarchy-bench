#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>

typedef struct {
    uint32_t min, max;
    uint64_t sum;
    uint32_t n;
} bench_stats_t;

void bench_stats(const uint32_t *v, uint32_t n, bench_stats_t *s);
void bench_print_ms(uint64_t cycles_x, uint32_t div);
void bench_print_us(uint64_t cycles_x, uint32_t div);
int bench_extmem_test(uintptr_t base, uint32_t bytes);

#endif
