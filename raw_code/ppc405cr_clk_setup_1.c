static void ppc405cr_clk_setup (ppc405cr_cpc_t *cpc)

{

    uint64_t VCO_out, PLL_out;

    uint32_t CPU_clk, TMR_clk, SDRAM_clk, PLB_clk, OPB_clk, EXT_clk, UART_clk;

    int M, D0, D1, D2;



    D0 = ((cpc->pllmr >> 26) & 0x3) + 1; /* CBDV */

    if (cpc->pllmr & 0x80000000) {

        D1 = (((cpc->pllmr >> 20) - 1) & 0xF) + 1; /* FBDV */

        D2 = 8 - ((cpc->pllmr >> 16) & 0x7); /* FWDVA */

        M = D0 * D1 * D2;

        VCO_out = cpc->sysclk * M;

        if (VCO_out < 400000000 || VCO_out > 800000000) {

            /* PLL cannot lock */

            cpc->pllmr &= ~0x80000000;

            goto bypass_pll;

        }

        PLL_out = VCO_out / D2;

    } else {

        /* Bypass PLL */

    bypass_pll:

        M = D0;

        PLL_out = cpc->sysclk * M;

    }

    CPU_clk = PLL_out;

    if (cpc->cr1 & 0x00800000)

        TMR_clk = cpc->sysclk; /* Should have a separate clock */

    else

        TMR_clk = CPU_clk;

    PLB_clk = CPU_clk / D0;

    SDRAM_clk = PLB_clk;

    D0 = ((cpc->pllmr >> 10) & 0x3) + 1;

    OPB_clk = PLB_clk / D0;

    D0 = ((cpc->pllmr >> 24) & 0x3) + 2;

    EXT_clk = PLB_clk / D0;

    D0 = ((cpc->cr0 >> 1) & 0x1F) + 1;

    UART_clk = CPU_clk / D0;

    /* Setup CPU clocks */

    clk_setup(&cpc->clk_setup[PPC405CR_CPU_CLK], CPU_clk);

    /* Setup time-base clock */

    clk_setup(&cpc->clk_setup[PPC405CR_TMR_CLK], TMR_clk);

    /* Setup PLB clock */

    clk_setup(&cpc->clk_setup[PPC405CR_PLB_CLK], PLB_clk);

    /* Setup SDRAM clock */

    clk_setup(&cpc->clk_setup[PPC405CR_SDRAM_CLK], SDRAM_clk);

    /* Setup OPB clock */

    clk_setup(&cpc->clk_setup[PPC405CR_OPB_CLK], OPB_clk);

    /* Setup external clock */

    clk_setup(&cpc->clk_setup[PPC405CR_EXT_CLK], EXT_clk);

    /* Setup UART clock */

    clk_setup(&cpc->clk_setup[PPC405CR_UART_CLK], UART_clk);

}
