#include "port.h"
#include "bench.h"
#include "nn_engine.h"
#include <stdio.h>

#ifndef BENCH_LABEL
#define BENCH_LABEL "unnamed"
#endif
#ifndef N_WARMUP
#define N_WARMUP 10u
#endif
#ifndef N_RUNS
#define N_RUNS 500u
#endif
#ifndef EXTMEM_TEST_BYTES
#define EXTMEM_TEST_BYTES (4u * 1024u * 1024u)
#endif

static uint32_t cyc[N_RUNS];

int main(void)
{
    port_init();

    uint32_t hz = port_cpu_hz();
    uint32_t ext_test = port_extmem_bytes() < EXTMEM_TEST_BYTES ? port_extmem_bytes() : EXTMEM_TEST_BYTES;

    printf("\n\n=== placement_sweep  %s  model %s  SRAM budget %lu KB ===\n",
           port_name(), NN_MODEL_NAME, (unsigned long)NN_PLAN_SRAM_KB);
    printf("CPU %lu MHz, ms = cycles / %lu\n", (unsigned long)(hz / 1000000u), (unsigned long)(hz / 1000u));
    printf("activations: SRAM arena %lu B, external extent %lu B at 0x%08lX, buffers external %lu/%u\n",
           (unsigned long)(NN_PLAN_SRAM_KB ? NN_SRAM_ARENA_BYTES : 0u), (unsigned long)NN_SDRAM_EXTENT,
           (unsigned long)port_extmem_base(), (unsigned long)NN_N_SDRAM_BUFS, NN_N_BUFS);

    if (port_extmem_init() != 0) { printf("external memory init FAIL\n"); port_halt(); }
    int t = bench_extmem_test(port_extmem_base(), ext_test);
    if (t != 0) { printf("external memory test FAIL at word %d\n", -t - 1); port_halt(); }
    printf("external memory init + test OK (%lu B at 0x%08lX)\n", (unsigned long)ext_test, (unsigned long)port_extmem_base());

    port_memory_config();
    printf("memory config: %s\n", port_memory_desc());

    if (nn_init() != 0) { printf("scratch too small\n"); port_halt(); }
    nn_load_input();
    int bad = nn_verify(1);
    uint32_t ref_ck = nn_checksum();
    printf("verify vs TFLite: %s (%d of %d layers differ), checksum 0x%08lX, argmax %d\n",
           bad ? "FAIL" : "all layers match", bad, NN_N_LAYERS, (unsigned long)ref_ck, nn_argmax());

    for (uint32_t i = 0; i < N_WARMUP; i++) { nn_load_input(); nn_run(); }
    printf("warm-up %lu runs done, timing %lu runs...\n", (unsigned long)N_WARMUP, (unsigned long)N_RUNS);
    port_flush();

    uint32_t ck_ok = 1u;
    for (uint32_t r = 0; r < N_RUNS; r++) {
        nn_load_input();
        uint32_t t0 = port_cycles();
        nn_run();
        cyc[r] = port_cycles() - t0;
        if (nn_checksum() != ref_ck) ck_ok = 0u;
    }

    printf("\nrun,cycles\n");
    for (uint32_t r = 0; r < N_RUNS; r++) printf("%lu,%lu\n", (unsigned long)(r + 1u), (unsigned long)cyc[r]);

    bench_stats_t s;
    bench_stats(cyc, N_RUNS, &s);
    printf("\nmin ");
    bench_print_ms(s.min, 1u);
    printf(" ms  mean ");
    bench_print_ms(s.sum, N_RUNS);
    printf(" ms  max ");
    bench_print_ms(s.max, 1u);
    printf(" ms  checksum 0x%08lX %s\n", (unsigned long)ref_ck, ck_ok ? "(same every run)" : "(CHANGED during runs!)");

    printf("\nconfig,board,model,sram_budget_kb,act_sram_kb,act_ext_kb,min_ms,mean_ms,max_ms,out_checksum\n");
    printf("CSV,%s,%s,%s,%lu,%lu,%lu,", BENCH_LABEL, port_name(), NN_MODEL_NAME, (unsigned long)NN_PLAN_SRAM_KB,
           (unsigned long)(NN_PLAN_SRAM_KB ? NN_SRAM_ARENA_BYTES / 1024u : 0u), (unsigned long)(NN_SDRAM_EXTENT / 1024u));
    bench_print_ms(s.min, 1u);
    printf(",");
    bench_print_ms(s.sum, N_RUNS);
    printf(",");
    bench_print_ms(s.max, 1u);
    printf(",0x%08lX\n\ndone\n", (unsigned long)ref_ck);
    port_flush();
    port_halt();
}
