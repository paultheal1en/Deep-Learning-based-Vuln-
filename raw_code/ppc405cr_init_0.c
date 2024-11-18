CPUState *ppc405cr_init (target_phys_addr_t ram_bases[4],

                         target_phys_addr_t ram_sizes[4],

                         uint32_t sysclk, qemu_irq **picp,

                         int do_init)

{

    clk_setup_t clk_setup[PPC405CR_CLK_NB];

    qemu_irq dma_irqs[4];

    CPUState *env;

    qemu_irq *pic, *irqs;



    memset(clk_setup, 0, sizeof(clk_setup));

    env = ppc4xx_init("405cr", &clk_setup[PPC405CR_CPU_CLK],

                      &clk_setup[PPC405CR_TMR_CLK], sysclk);

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

    ppc4xx_sdram_init(env, pic[14], 1, ram_bases, ram_sizes, do_init);

    /* External bus controller */

    ppc405_ebc_init(env);

    /* DMA controller */

    dma_irqs[0] = pic[26];

    dma_irqs[1] = pic[25];

    dma_irqs[2] = pic[24];

    dma_irqs[3] = pic[23];

    ppc405_dma_init(env, dma_irqs);

    /* Serial ports */

    if (serial_hds[0] != NULL) {

        serial_mm_init(0xef600300, 0, pic[0], PPC_SERIAL_MM_BAUDBASE,

                       serial_hds[0], 1, 1);

    }

    if (serial_hds[1] != NULL) {

        serial_mm_init(0xef600400, 0, pic[1], PPC_SERIAL_MM_BAUDBASE,

                       serial_hds[1], 1, 1);

    }

    /* IIC controller */

    ppc405_i2c_init(0xef600500, pic[2]);

    /* GPIO */

    ppc405_gpio_init(0xef600700);

    /* CPU control */

    ppc405cr_cpc_init(env, clk_setup, sysclk);



    return env;

}
