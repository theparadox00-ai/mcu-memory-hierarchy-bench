#include "stm32h7xx.h"
#include "port.h"
#include "port_overheads.h"

#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define OV_BYTES (128u * 1024u)

__attribute__((aligned(131072))) static uint8_t axi_buf[OV_BYTES];

void *port_ov_buffer(uint32_t *bytes)
{
    *bytes = OV_BYTES;
    return axi_buf;
}

static void mpu_test_region(void)
{
    MPU->RNR = 1u;
    MPU->RBAR = (uint32_t)axi_buf;
    MPU->RASR = (1u << MPU_RASR_XN_Pos) | (3u << MPU_RASR_AP_Pos) |
                (1u << MPU_RASR_TEX_Pos) | (0u << MPU_RASR_S_Pos) |
                (1u << MPU_RASR_C_Pos) | (1u << MPU_RASR_B_Pos) |
                (16u << MPU_RASR_SIZE_Pos) | MPU_RASR_ENABLE_Msk;
}

uint32_t port_ov_mpu_reprogram(void)
{
    uint32_t t0 = DWT_CYCCNT;
    __DMB();
    MPU->CTRL = 0u;
    mpu_test_region();
    MPU->CTRL = MPU_CTRL_PRIVDEFENA_Msk | MPU_CTRL_ENABLE_Msk;
    __DSB();
    __ISB();
    return DWT_CYCCNT - t0;
}

int port_ov_dma_init(void)
{
    RCC->AHB3ENR |= RCC_AHB3ENR_MDMAEN;
    (void)RCC->AHB3ENR;
    MDMA_Channel0->CCR = 0u;
    while (MDMA_Channel0->CCR & MDMA_CCR_EN) {}
    return 0;
}

uint32_t port_ov_dma_copy(const void *src, void *dst, uint32_t bytes)
{
    MDMA_Channel_TypeDef *ch = MDMA_Channel0;
    ch->CCR = 0u;
    ch->CIFCR = 0x1Fu;
    ch->CTCR = (2u << MDMA_CTCR_SINC_Pos) | (2u << MDMA_CTCR_DINC_Pos) |
               (3u << MDMA_CTCR_SSIZE_Pos) | (3u << MDMA_CTCR_DSIZE_Pos) |
               (3u << MDMA_CTCR_SINCOS_Pos) | (3u << MDMA_CTCR_DINCOS_Pos) |
               (4u << MDMA_CTCR_SBURST_Pos) | (4u << MDMA_CTCR_DBURST_Pos) |
               (127u << MDMA_CTCR_TLEN_Pos) | (2u << MDMA_CTCR_TRGM_Pos) | MDMA_CTCR_SWRM;
    ch->CBNDTR = ((bytes / 4u) << MDMA_CBNDTR_BNDT_Pos) | (3u << MDMA_CBNDTR_BRC_Pos);
    ch->CSAR = (uint32_t)src;
    ch->CDAR = (uint32_t)dst;
    ch->CBRUR = 0u;
    ch->CLAR = 0u;
    ch->CTBR = 0u;
    ch->CMAR = 0u;
    ch->CMDR = 0u;
    ch->CCR = (3u << MDMA_CCR_PL_Pos) | MDMA_CCR_EN;
    __DSB();
    uint32_t t0 = DWT_CYCCNT;
    ch->CCR |= MDMA_CCR_SWRQ;
    while ((ch->CISR & (MDMA_CISR_CTCIF | MDMA_CISR_TEIF)) == 0u) {}
    uint32_t t = DWT_CYCCNT - t0;
    if (ch->CISR & MDMA_CISR_TEIF) t = PORT_OV_FAIL;
    ch->CIFCR = 0x1Fu;
    ch->CCR = 0u;
    return t;
}

uint32_t port_ov_cache_clean_invalidate(void *p, uint32_t bytes)
{
    __DSB();
    uint32_t t0 = DWT_CYCCNT;
    SCB_CleanInvalidateDCache_by_Addr(p, (int32_t)bytes);
    return DWT_CYCCNT - t0;
}

void port_ov_cache_clean(void *p, uint32_t bytes)
{
    SCB_CleanDCache_by_Addr(p, (int32_t)bytes);
}

void port_ov_cache_invalidate(void *p, uint32_t bytes)
{
    SCB_InvalidateDCache_by_Addr(p, (int32_t)bytes);
}
