#include "stm32h7xx.h"
#include "port.h"
#include <stdio.h>

#define DEMCR      (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define DWT_LAR    (*(volatile uint32_t *)0xE0001FB0u)

#define CPU_HZ 400000000u
#define EXTMEM_BASE 0xD0000000u
#define EXTMEM_BYTES (32u * 1024u * 1024u)
#define DTCM_BASE 0x20000000u
#define DTCM_BYTES (128u * 1024u)
#ifndef EXTMEM_CACHEABLE
#define EXTMEM_CACHEABLE 1
#endif

static void uart_init(uint32_t bus_hz)
{
    RCC->AHB4ENR |= RCC_AHB4ENR_GPIOBEN;
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN;
    (void)RCC->APB2ENR;
    GPIOB->MODER &= ~((3u << (14u * 2u)) | (3u << (15u * 2u)));
    GPIOB->MODER |=  ((2u << (14u * 2u)) | (2u << (15u * 2u)));
    GPIOB->AFR[1] &= ~((0xFu << ((14u - 8u) * 4u)) | (0xFu << ((15u - 8u) * 4u)));
    GPIOB->AFR[1] |=  ((4u << ((14u - 8u) * 4u)) | (4u << ((15u - 8u) * 4u)));
    USART1->CR1 = 0u;
    USART1->BRR = bus_hz / 115200u;
    USART1->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

static void uart_putc(char c)
{
    while ((USART1->ISR & USART_ISR_TXE_TXFNF) == 0u) {}
    USART1->TDR = (uint8_t)c;
}

static void uart_wait_done(void)
{
    while ((USART1->ISR & USART_ISR_TC) == 0u) {}
}

static void raw_puts(const char *p)
{
    while (*p) { if (*p == '\n') uart_putc('\r'); uart_putc(*p++); }
}

static void raw_hex(uint32_t v)
{
    static const char h[] = "0123456789ABCDEF";
    uart_putc('0'); uart_putc('x');
    for (int i = 28; i >= 0; i -= 4) uart_putc(h[(v >> i) & 0xFu]);
}

void fault_report(uint32_t *sp)
{
    raw_puts("\n\n*** HARD FAULT ***\nPC   = "); raw_hex(sp[6]);
    raw_puts("\nLR   = "); raw_hex(sp[5]);
    raw_puts("\nCFSR = "); raw_hex(SCB->CFSR);
    raw_puts("\nHFSR = "); raw_hex(SCB->HFSR);
    raw_puts("\nMMFAR= "); raw_hex(SCB->MMFAR);
    raw_puts("\nBFAR = "); raw_hex(SCB->BFAR);
    raw_puts("\n");
    while (1) {}
}

__attribute__((naked)) void HardFault_Handler(void)
{
    __asm volatile(
        "tst lr, #4      \n"
        "ite eq          \n"
        "mrseq r0, msp   \n"
        "mrsne r0, psp   \n"
        "b fault_report  \n");
}

__attribute__((naked)) void MemManage_Handler(void)  { __asm volatile("b HardFault_Handler"); }
__attribute__((naked)) void BusFault_Handler(void)   { __asm volatile("b HardFault_Handler"); }
__attribute__((naked)) void UsageFault_Handler(void) { __asm volatile("b HardFault_Handler"); }

static void clock_init(void)
{
    PWR->CR3 = (PWR->CR3 & ~(PWR_CR3_LDOEN | PWR_CR3_BYPASS)) | PWR_CR3_SMPSEN;
    while ((PWR->CSR1 & PWR_CSR1_ACTVOSRDY) == 0u) {}
    PWR->D3CR = (PWR->D3CR & ~PWR_D3CR_VOS) | PWR_D3CR_VOS;
    while ((PWR->D3CR & PWR_D3CR_VOSRDY) == 0u) {}
    FLASH->ACR = FLASH_ACR_LATENCY_2WS | FLASH_ACR_WRHIGHFREQ_1;
    while ((FLASH->ACR & FLASH_ACR_LATENCY) != FLASH_ACR_LATENCY_2WS) {}
    RCC->CR |= RCC_CR_HSEON;
    while ((RCC->CR & RCC_CR_HSERDY) == 0u) {}
    RCC->PLLCKSELR = RCC_PLLCKSELR_PLLSRC_HSE | (5u << RCC_PLLCKSELR_DIVM1_Pos);
    RCC->PLLCFGR = (RCC->PLLCFGR & ~(RCC_PLLCFGR_PLL1RGE | RCC_PLLCFGR_PLL1VCOSEL | RCC_PLLCFGR_PLL1FRACEN))
                 | (2u << RCC_PLLCFGR_PLL1RGE_Pos)
                 | RCC_PLLCFGR_DIVP1EN | RCC_PLLCFGR_DIVQ1EN | RCC_PLLCFGR_DIVR1EN;
    RCC->PLL1DIVR = ((160u - 1u) << RCC_PLL1DIVR_N1_Pos)
                  | ((2u - 1u)   << RCC_PLL1DIVR_P1_Pos)
                  | ((4u - 1u)   << RCC_PLL1DIVR_Q1_Pos)
                  | ((2u - 1u)   << RCC_PLL1DIVR_R1_Pos);
    RCC->CR |= RCC_CR_PLL1ON;
    while ((RCC->CR & RCC_CR_PLL1RDY) == 0u) {}
    uart_wait_done();
    RCC->D1CFGR = RCC_D1CFGR_HPRE_DIV2 | RCC_D1CFGR_D1PPRE_DIV2;
    RCC->D2CFGR = RCC_D2CFGR_D2PPRE1_DIV2 | RCC_D2CFGR_D2PPRE2_DIV2;
    RCC->D3CFGR = RCC_D3CFGR_D3PPRE_DIV2;
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL1;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL1) {}
    uart_init(100000000u);
}

static void dwt_init(void)
{
    DEMCR |= (1u << 24);
    DWT_LAR = 0xC5ACCE55u;
    DWT_CYCCNT = 0u;
    DWT_CTRL |= 1u;
}

static void delay_cycles(uint32_t n)
{
    uint32_t s = DWT_CYCCNT;
    while ((DWT_CYCCNT - s) < n) {}
}

static void pin_af(GPIO_TypeDef *port, uint32_t pin, uint32_t af, uint32_t pullup)
{
    port->MODER   = (port->MODER   & ~(3u << (pin * 2u))) | (2u << (pin * 2u));
    port->OSPEEDR |= (3u << (pin * 2u));
    port->OTYPER  &= ~(1u << pin);
    port->PUPDR   = (port->PUPDR   & ~(3u << (pin * 2u))) | ((pullup ? 1u : 0u) << (pin * 2u));
    if (pin < 8u) port->AFR[0] = (port->AFR[0] & ~(0xFu << (pin * 4u))) | (af << (pin * 4u));
    else          port->AFR[1] = (port->AFR[1] & ~(0xFu << ((pin - 8u) * 4u))) | (af << ((pin - 8u) * 4u));
}

static void sdram_wait(void)
{
    while (FMC_Bank5_6_R->SDSR & (1u << 5)) {}
}

static void sdram_cmd(uint32_t mode, uint32_t nrfs, uint32_t mrd)
{
    sdram_wait();
    FMC_Bank5_6_R->SDCMR = (mode << FMC_SDCMR_MODE_Pos) | (1u << FMC_SDCMR_CTB2_Pos) |
                           ((nrfs - 1u) << FMC_SDCMR_NRFS_Pos) | (mrd << FMC_SDCMR_MRD_Pos);
    sdram_wait();
}

static void fmc_sdram_init(void)
{
    static const uint8_t pd[] = { 0, 1, 8, 9, 10, 14, 15 };
    static const uint8_t pe[] = { 0, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    static const uint8_t pf[] = { 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15 };
    static const uint8_t pg[] = { 0, 1, 2, 3, 4, 5, 8, 15 };
    static const uint8_t ph[] = { 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    static const uint8_t pi[] = { 0, 1, 2, 3, 4, 5, 6, 7, 9, 10 };

    RCC->AHB4ENR |= RCC_AHB4ENR_GPIODEN | RCC_AHB4ENR_GPIOEEN | RCC_AHB4ENR_GPIOFEN |
                    RCC_AHB4ENR_GPIOGEN | RCC_AHB4ENR_GPIOHEN | RCC_AHB4ENR_GPIOIEN;
    RCC->AHB3ENR |= RCC_AHB3ENR_FMCEN;
    (void)RCC->AHB3ENR;

    for (uint32_t i = 0; i < sizeof(pd); i++) pin_af(GPIOD, pd[i], 12u, 1u);
    for (uint32_t i = 0; i < sizeof(pe); i++) pin_af(GPIOE, pe[i], 12u, 1u);
    for (uint32_t i = 0; i < sizeof(pf); i++) pin_af(GPIOF, pf[i], 12u, 1u);
    for (uint32_t i = 0; i < sizeof(pg); i++) pin_af(GPIOG, pg[i], 12u, 1u);
    for (uint32_t i = 0; i < sizeof(ph); i++) pin_af(GPIOH, ph[i], 12u, 1u);
    for (uint32_t i = 0; i < sizeof(pi); i++) pin_af(GPIOI, pi[i], 12u, 1u);

    FMC_Bank5_6_R->SDCR[0] = (2u << FMC_SDCRx_SDCLK_Pos) | (1u << FMC_SDCRx_RBURST_Pos) | (0u << FMC_SDCRx_RPIPE_Pos);
    FMC_Bank5_6_R->SDCR[1] = (1u << FMC_SDCRx_NC_Pos) | (1u << FMC_SDCRx_NR_Pos) | (2u << FMC_SDCRx_MWID_Pos) |
                             (1u << FMC_SDCRx_NB_Pos) | (3u << FMC_SDCRx_CAS_Pos);
    FMC_Bank5_6_R->SDTR[0] = ((7u - 1u) << FMC_SDTRx_TRC_Pos) | ((2u - 1u) << FMC_SDTRx_TRP_Pos);
    FMC_Bank5_6_R->SDTR[1] = ((2u - 1u) << FMC_SDTRx_TMRD_Pos) | ((7u - 1u) << FMC_SDTRx_TXSR_Pos) |
                             ((4u - 1u) << FMC_SDTRx_TRAS_Pos) | ((2u - 1u) << FMC_SDTRx_TWR_Pos) |
                             ((2u - 1u) << FMC_SDTRx_TRCD_Pos);
    FMC_Bank1_R->BTCR[0] |= FMC_BCR1_FMCEN_Msk;

    sdram_cmd(1u, 1u, 0u);
    delay_cycles(400000u);
    sdram_cmd(2u, 1u, 0u);
    sdram_cmd(3u, 8u, 0u);
    sdram_cmd(4u, 1u, 0x230u);
    FMC_Bank5_6_R->SDRTR = (0x603u << FMC_SDRTR_COUNT_Pos);
}

static void mpu_init(void)
{
    __DMB();
    MPU->CTRL = 0u;
    MPU->RNR = 0u;
    MPU->RBAR = EXTMEM_BASE;
    MPU->RASR = (1u << MPU_RASR_XN_Pos) | (3u << MPU_RASR_AP_Pos) |
                (1u << MPU_RASR_TEX_Pos) | (0u << MPU_RASR_S_Pos) |
                ((uint32_t)EXTMEM_CACHEABLE << MPU_RASR_C_Pos) | ((uint32_t)EXTMEM_CACHEABLE << MPU_RASR_B_Pos) |
                (24u << MPU_RASR_SIZE_Pos) | MPU_RASR_ENABLE_Msk;
    MPU->CTRL = MPU_CTRL_PRIVDEFENA_Msk | MPU_CTRL_ENABLE_Msk;
    SCB->SHCSR |= SCB_SHCSR_MEMFAULTENA_Msk;
    __DSB();
    __ISB();
}

static void caches_on(void)
{
    SCB_InvalidateICache();
    SCB_EnableICache();
    SCB_InvalidateDCache();
    SCB_EnableDCache();
}

void port_init(void)
{
    SCB->CPACR |= (0xFu << 20);
    __DSB();
    __ISB();
    uart_init(64000000u);
    clock_init();
    dwt_init();
}

const char *port_name(void) { return "stm32h757i_eval"; }
uint32_t port_cpu_hz(void) { return CPU_HZ; }

uint32_t port_cycles(void)
{
    __DSB();
    return DWT_CYCCNT;
}

void port_putc(char c) { uart_putc(c); }
void port_flush(void) { uart_wait_done(); }

int port_extmem_init(void)
{
    fmc_sdram_init();
    return 0;
}

uintptr_t port_extmem_base(void) { return EXTMEM_BASE; }
uint32_t port_extmem_bytes(void) { return EXTMEM_BYTES; }

void port_memory_config(void)
{
    mpu_init();
    caches_on();
}

const char *port_memory_desc(void)
{
#if EXTMEM_CACHEABLE
    return "MPU region 0: 0xD0000000 32 MB Normal WB-RWA non-shareable XN; I/D cache ON";
#else
    return "MPU region 0: 0xD0000000 32 MB Normal non-cacheable non-shareable XN; I/D cache ON";
#endif
}

void *port_scratch(uint32_t *bytes)
{
    *bytes = DTCM_BYTES;
    return (void *)DTCM_BASE;
}

void port_halt(void)
{
    while (1) {}
}
