CPUState *ppc405ep_init (target_phys_addr_t ram_bases[2],

                         target_phys_addr_t ram_sizes[2],

                         uint32_t sysclk, qemu_irq **picp,

                         int do_init)

{

    clk_setup_t clk_setup[PPC405EP_CLK_NB], tlb_clk_setup;

    qemu_irq dma_irqs[4], gpt_irqs[5], mal_irqs[4];

    CPUState *env;

    qemu_irq *pic, *irqs;



    memset(clk_setup, 0, sizeof(clk_setup));

    /* init CPUs */

    env = ppc4xx_init("405ep", &clk_setup[PPC405EP_CPU_CLK],

                      &tlb_clk_setup, sysclk);

    clk_setup[PPC405EP_CPU_CLK].cb = tlb_clk_setup.cb;

    clk_setup[PPC405EP_CPU_CLK].opaque = tlb_clk_setup.opaque;

    /* Internal devices init */

    /* Memory mapped devices registers */

    /* PLB arbitrer */

    ppc4xx_plb_init(env);

    /* PLB to OPB bridge */

    ppc4xx_pob_init(env);

    /* OBP arbitrer */

    ppc4xx_opba_init(0xef600600);

    /* Universal interrupt controller */

    irqs = g_malloc0(sizeof(qemu_irq) * PPCUIC_OUTPUT_NB);

    irqs[PPCUIC_OUTPUT_INT] =

        ((qemu_irq *)env->irq_inputs)[PPC40x_INPUT_INT];

    irqs[PPCUIC_OUTPUT_CINT] =

        ((qemu_irq *)env->irq_inputs)[PPC40x_INPUT_CINT];

    pic = ppcuic_init(env, irqs, 0x0C0, 0, 1);

    *picp = pic;

    /* SDRAM controller */

	/* XXX 405EP has no ECC interrupt */

    ppc4xx_sdram_init(env, pic[17], 2, ram_bases, ram_sizes, do_init);

    /* External bus controller */

    ppc405_ebc_init(env);

    /* DMA controller */

    dma_irqs[0] = pic[5];

    dma_irqs[1] = pic[6];

    dma_irqs[2] = pic[7];

    dma_irqs[3] = pic[8];

    ppc405_dma_init(env, dma_irqs);

    /* IIC controller */

    ppc405_i2c_init(0xef600500, pic[2]);

    /* GPIO */

    ppc405_gpio_init(0xef600700);

    /* Serial ports */

    if (serial_hds[0] != NULL) {

        serial_mm_init(0xef600300, 0, pic[0], PPC_SERIAL_MM_BAUDBASE,

                       serial_hds[0], 1, 1);

    }

    if (serial_hds[1] != NULL) {

        serial_mm_init(0xef600400, 0, pic[1], PPC_SERIAL_MM_BAUDBASE,

                       serial_hds[1], 1, 1);

    }

    /* OCM */

    ppc405_ocm_init(env);

    /* GPT */

    gpt_irqs[0] = pic[19];

    gpt_irqs[1] = pic[20];

    gpt_irqs[2] = pic[21];

    gpt_irqs[3] = pic[22];

    gpt_irqs[4] = pic[23];

    ppc4xx_gpt_init(0xef600000, gpt_irqs);

    /* PCI */

    /* Uses pic[3], pic[16], pic[18] */

    /* MAL */

    mal_irqs[0] = pic[11];

    mal_irqs[1] = pic[12];

    mal_irqs[2] = pic[13];

    mal_irqs[3] = pic[14];

    ppc405_mal_init(env, mal_irqs);

    /* Ethernet */

    /* Uses pic[9], pic[15], pic[17] */

    /* CPU control */

    ppc405ep_cpc_init(env, clk_setup, sysclk);



    return env;

}
