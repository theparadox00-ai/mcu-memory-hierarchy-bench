#include "port.h"
#include "port_overheads.h"
#include "bench.h"
#include <stdio.h>

#ifndef N_WARMUP
#define N_WARMUP 10u
#endif
#ifndef N_ITER
#define N_ITER 1000u
#endif
#ifndef EXTMEM_TEST_BYTES
#define EXTMEM_TEST_BYTES (4u * 1024u * 1024u)
#endif
#ifndef DMA_DST_OFFSET
#define DMA_DST_OFFSET 0x00100000u
#endif

static uint32_t t_mpu[N_ITER], t_dc[N_ITER], t_dma[N_ITER];

static void report(const char *name, const uint32_t *t, uint32_t bytes)
{
    bench_stats_t s;
    bench_stats(t, N_ITER, &s);
    printf("CSV,%s,%s,", port_name(), name);
    bench_print_us(s.sum, N_ITER);
    printf(",");
    bench_print_us(s.max, 1u);
    printf(",");
    bench_print_us(s.min, 1u);
    printf(",%lu,", (unsigned long)(s.sum / N_ITER));
    if (bytes) {
        uint64_t kbps = (uint64_t)bytes * port_cpu_hz() / 1000u * N_ITER / s.sum;
        printf("%lu.%03lu", (unsigned long)(kbps / 1000u), (unsigned long)(kbps % 1000u));
    }
    printf("\n");
}

static void fill(uint32_t *p, uint32_t words, uint32_t seed, int hashed)
{
    for (uint32_t k = 0; k < words; k++) p[k] = hashed ? ((k * 2654435761u) ^ seed) : (k ^ seed);
}

int main(void)
{
    port_init();

    uint32_t bytes = 0u;
    uint8_t *buf = (uint8_t *)port_ov_buffer(&bytes);
    uint32_t words = bytes / 4u;
    uint32_t ext_test = port_extmem_bytes() < EXTMEM_TEST_BYTES ? port_extmem_bytes() : EXTMEM_TEST_BYTES;

    printf("\n\n=== overhead_bench  %s ===\n", port_name());
    printf("CPU %lu MHz, us = cycles / %lu, %lu iterations each, buffer %lu B at 0x%08lX\n",
           (unsigned long)(port_cpu_hz() / 1000000u), (unsigned long)(port_cpu_hz() / 1000000u),
           (unsigned long)N_ITER, (unsigned long)bytes, (unsigned long)(uintptr_t)buf);

    if (port_extmem_init() != 0) { printf("external memory init FAIL\n"); port_halt(); }
    int t = bench_extmem_test(port_extmem_base(), ext_test);
    if (t != 0) { printf("external memory test FAIL at word %d\n", -t - 1); port_halt(); }
    printf("external memory init + test OK (%lu B at 0x%08lX)\n", (unsigned long)ext_test, (unsigned long)port_extmem_base());

    port_memory_config();
    printf("memory config: %s\n", port_memory_desc());

    uint8_t *dst = (uint8_t *)(port_extmem_base() + DMA_DST_OFFSET);
    if (port_ov_dma_init() != 0) { printf("DMA init FAIL\n"); port_halt(); }

    for (uint32_t i = 0; i < N_WARMUP; i++) (void)port_ov_mpu_reprogram();
    for (uint32_t i = 0; i < N_ITER; i++) t_mpu[i] = port_ov_mpu_reprogram();

    for (uint32_t i = 0; i < N_ITER + N_WARMUP; i++) {
        fill((uint32_t *)buf, words, i, 0);
        uint32_t c = port_ov_cache_clean_invalidate(buf, bytes);
        if (i >= N_WARMUP) t_dc[i - N_WARMUP] = c;
    }

    uint32_t bad = 0u, err = 0u;
    for (uint32_t i = 0; i < N_ITER + N_WARMUP; i++) {
        fill((uint32_t *)buf, words, i, 1);
        port_ov_cache_clean(buf, bytes);
        uint32_t c = port_ov_dma_copy(buf, dst, bytes);
        if (c == PORT_OV_FAIL) err++;
        if (i >= N_WARMUP) t_dma[i - N_WARMUP] = c;
        if (i == 0u || i == N_ITER + N_WARMUP - 1u) {
            port_ov_cache_invalidate(dst, bytes);
            const uint32_t *q = (const uint32_t *)dst;
            for (uint32_t k = 0; k < words; k++) if (q[k] != ((k * 2654435761u) ^ i)) bad++;
        }
    }
    printf("DMA: transfer errors %lu, copy check %s (%lu words wrong)\n",
           (unsigned long)err, bad ? "FAIL" : "OK", (unsigned long)bad);

    printf("\nboard,test,mean_us,max_us,min_us,mean_cycles,MB_per_s\n");
    report("mpu_reprogram", t_mpu, 0u);
    report("dcache_clean_invalidate", t_dc, 0u);
    report("dma_copy_fast_to_ext", t_dma, bytes);
    printf("\ndone\n");
    port_flush();
    port_halt();
}
