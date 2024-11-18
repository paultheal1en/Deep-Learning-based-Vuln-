struct omap_mpu_state_s *omap310_mpu_init(MemoryRegion *system_memory,

                unsigned long sdram_size,

                const char *core)

{

    int i;

    struct omap_mpu_state_s *s = (struct omap_mpu_state_s *)

            g_malloc0(sizeof(struct omap_mpu_state_s));

    qemu_irq dma_irqs[6];

    DriveInfo *dinfo;

    SysBusDevice *busdev;



    if (!core)

        core = "ti925t";



    /* Core */

    s->mpu_model = omap310;

    s->cpu = cpu_arm_init(core);

    if (s->cpu == NULL) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    s->sdram_size = sdram_size;

    s->sram_size = OMAP15XX_SRAM_SIZE;



    s->wakeup = qemu_allocate_irq(omap_mpu_wakeup, s, 0);



    /* Clocks */

    omap_clk_init(s);



    /* Memory-mapped stuff */

    memory_region_allocate_system_memory(&s->emiff_ram, NULL, "omap1.dram",

                                         s->sdram_size);

    memory_region_add_subregion(system_memory, OMAP_EMIFF_BASE, &s->emiff_ram);

    memory_region_init_ram(&s->imif_ram, NULL, "omap1.sram", s->sram_size,

                           &error_abort);

    vmstate_register_ram_global(&s->imif_ram);

    memory_region_add_subregion(system_memory, OMAP_IMIF_BASE, &s->imif_ram);



    omap_clkm_init(system_memory, 0xfffece00, 0xe1008000, s);



    s->ih[0] = qdev_create(NULL, "omap-intc");

    qdev_prop_set_uint32(s->ih[0], "size", 0x100);

    qdev_prop_set_ptr(s->ih[0], "clk", omap_findclk(s, "arminth_ck"));

    qdev_init_nofail(s->ih[0]);

    busdev = SYS_BUS_DEVICE(s->ih[0]);

    sysbus_connect_irq(busdev, 0,

                       qdev_get_gpio_in(DEVICE(s->cpu), ARM_CPU_IRQ));

    sysbus_connect_irq(busdev, 1,

                       qdev_get_gpio_in(DEVICE(s->cpu), ARM_CPU_FIQ));

    sysbus_mmio_map(busdev, 0, 0xfffecb00);

    s->ih[1] = qdev_create(NULL, "omap-intc");

    qdev_prop_set_uint32(s->ih[1], "size", 0x800);

    qdev_prop_set_ptr(s->ih[1], "clk", omap_findclk(s, "arminth_ck"));

    qdev_init_nofail(s->ih[1]);

    busdev = SYS_BUS_DEVICE(s->ih[1]);

    sysbus_connect_irq(busdev, 0,

                       qdev_get_gpio_in(s->ih[0], OMAP_INT_15XX_IH2_IRQ));

    /* The second interrupt controller's FIQ output is not wired up */

    sysbus_mmio_map(busdev, 0, 0xfffe0000);



    for (i = 0; i < 6; i++) {

        dma_irqs[i] = qdev_get_gpio_in(s->ih[omap1_dma_irq_map[i].ih],

                                       omap1_dma_irq_map[i].intr);

    }

    s->dma = omap_dma_init(0xfffed800, dma_irqs, system_memory,

                           qdev_get_gpio_in(s->ih[0], OMAP_INT_DMA_LCD),

                           s, omap_findclk(s, "dma_ck"), omap_dma_3_1);



    s->port[emiff    ].addr_valid = omap_validate_emiff_addr;

    s->port[emifs    ].addr_valid = omap_validate_emifs_addr;

    s->port[imif     ].addr_valid = omap_validate_imif_addr;

    s->port[tipb     ].addr_valid = omap_validate_tipb_addr;

    s->port[local    ].addr_valid = omap_validate_local_addr;

    s->port[tipb_mpui].addr_valid = omap_validate_tipb_mpui_addr;



    /* Register SDRAM and SRAM DMA ports for fast transfers.  */

    soc_dma_port_add_mem(s->dma, memory_region_get_ram_ptr(&s->emiff_ram),

                         OMAP_EMIFF_BASE, s->sdram_size);

    soc_dma_port_add_mem(s->dma, memory_region_get_ram_ptr(&s->imif_ram),

                         OMAP_IMIF_BASE, s->sram_size);



    s->timer[0] = omap_mpu_timer_init(system_memory, 0xfffec500,

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_TIMER1),

                    omap_findclk(s, "mputim_ck"));

    s->timer[1] = omap_mpu_timer_init(system_memory, 0xfffec600,

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_TIMER2),

                    omap_findclk(s, "mputim_ck"));

    s->timer[2] = omap_mpu_timer_init(system_memory, 0xfffec700,

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_TIMER3),

                    omap_findclk(s, "mputim_ck"));



    s->wdt = omap_wd_timer_init(system_memory, 0xfffec800,

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_WD_TIMER),

                    omap_findclk(s, "armwdt_ck"));



    s->os_timer = omap_os_timer_init(system_memory, 0xfffb9000,

                    qdev_get_gpio_in(s->ih[1], OMAP_INT_OS_TIMER),

                    omap_findclk(s, "clk32-kHz"));



    s->lcd = omap_lcdc_init(system_memory, 0xfffec000,

                            qdev_get_gpio_in(s->ih[0], OMAP_INT_LCD_CTRL),

                            omap_dma_get_lcdch(s->dma),

                            omap_findclk(s, "lcd_ck"));



    omap_ulpd_pm_init(system_memory, 0xfffe0800, s);

    omap_pin_cfg_init(system_memory, 0xfffe1000, s);

    omap_id_init(system_memory, s);



    omap_mpui_init(system_memory, 0xfffec900, s);



    s->private_tipb = omap_tipb_bridge_init(system_memory, 0xfffeca00,

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_BRIDGE_PRIV),

                    omap_findclk(s, "tipb_ck"));

    s->public_tipb = omap_tipb_bridge_init(system_memory, 0xfffed300,

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_BRIDGE_PUB),

                    omap_findclk(s, "tipb_ck"));



    omap_tcmi_init(system_memory, 0xfffecc00, s);



    s->uart[0] = omap_uart_init(0xfffb0000,

                                qdev_get_gpio_in(s->ih[1], OMAP_INT_UART1),

                    omap_findclk(s, "uart1_ck"),

                    omap_findclk(s, "uart1_ck"),

                    s->drq[OMAP_DMA_UART1_TX], s->drq[OMAP_DMA_UART1_RX],

                    "uart1",

                    serial_hds[0]);

    s->uart[1] = omap_uart_init(0xfffb0800,

                                qdev_get_gpio_in(s->ih[1], OMAP_INT_UART2),

                    omap_findclk(s, "uart2_ck"),

                    omap_findclk(s, "uart2_ck"),

                    s->drq[OMAP_DMA_UART2_TX], s->drq[OMAP_DMA_UART2_RX],

                    "uart2",

                    serial_hds[0] ? serial_hds[1] : NULL);

    s->uart[2] = omap_uart_init(0xfffb9800,

                                qdev_get_gpio_in(s->ih[0], OMAP_INT_UART3),

                    omap_findclk(s, "uart3_ck"),

                    omap_findclk(s, "uart3_ck"),

                    s->drq[OMAP_DMA_UART3_TX], s->drq[OMAP_DMA_UART3_RX],

                    "uart3",

                    serial_hds[0] && serial_hds[1] ? serial_hds[2] : NULL);



    s->dpll[0] = omap_dpll_init(system_memory, 0xfffecf00,

                                omap_findclk(s, "dpll1"));

    s->dpll[1] = omap_dpll_init(system_memory, 0xfffed000,

                                omap_findclk(s, "dpll2"));

    s->dpll[2] = omap_dpll_init(system_memory, 0xfffed100,

                                omap_findclk(s, "dpll3"));



    dinfo = drive_get(IF_SD, 0, 0);

    if (!dinfo) {

        fprintf(stderr, "qemu: missing SecureDigital device\n");

        exit(1);

    }

    s->mmc = omap_mmc_init(0xfffb7800, system_memory,

                           blk_by_legacy_dinfo(dinfo),

                           qdev_get_gpio_in(s->ih[1], OMAP_INT_OQN),

                           &s->drq[OMAP_DMA_MMC_TX],

                    omap_findclk(s, "mmc_ck"));



    s->mpuio = omap_mpuio_init(system_memory, 0xfffb5000,

                               qdev_get_gpio_in(s->ih[1], OMAP_INT_KEYBOARD),

                               qdev_get_gpio_in(s->ih[1], OMAP_INT_MPUIO),

                               s->wakeup, omap_findclk(s, "clk32-kHz"));



    s->gpio = qdev_create(NULL, "omap-gpio");

    qdev_prop_set_int32(s->gpio, "mpu_model", s->mpu_model);

    qdev_prop_set_ptr(s->gpio, "clk", omap_findclk(s, "arm_gpio_ck"));

    qdev_init_nofail(s->gpio);

    sysbus_connect_irq(SYS_BUS_DEVICE(s->gpio), 0,

                       qdev_get_gpio_in(s->ih[0], OMAP_INT_GPIO_BANK1));

    sysbus_mmio_map(SYS_BUS_DEVICE(s->gpio), 0, 0xfffce000);



    s->microwire = omap_uwire_init(system_memory, 0xfffb3000,

                                   qdev_get_gpio_in(s->ih[1], OMAP_INT_uWireTX),

                                   qdev_get_gpio_in(s->ih[1], OMAP_INT_uWireRX),

                    s->drq[OMAP_DMA_UWIRE_TX], omap_findclk(s, "mpuper_ck"));



    s->pwl = omap_pwl_init(system_memory, 0xfffb5800,

                           omap_findclk(s, "armxor_ck"));

    s->pwt = omap_pwt_init(system_memory, 0xfffb6000,

                           omap_findclk(s, "armxor_ck"));



    s->i2c[0] = qdev_create(NULL, "omap_i2c");

    qdev_prop_set_uint8(s->i2c[0], "revision", 0x11);

    qdev_prop_set_ptr(s->i2c[0], "fclk", omap_findclk(s, "mpuper_ck"));

    qdev_init_nofail(s->i2c[0]);

    busdev = SYS_BUS_DEVICE(s->i2c[0]);

    sysbus_connect_irq(busdev, 0, qdev_get_gpio_in(s->ih[1], OMAP_INT_I2C));

    sysbus_connect_irq(busdev, 1, s->drq[OMAP_DMA_I2C_TX]);

    sysbus_connect_irq(busdev, 2, s->drq[OMAP_DMA_I2C_RX]);

    sysbus_mmio_map(busdev, 0, 0xfffb3800);



    s->rtc = omap_rtc_init(system_memory, 0xfffb4800,

                           qdev_get_gpio_in(s->ih[1], OMAP_INT_RTC_TIMER),

                           qdev_get_gpio_in(s->ih[1], OMAP_INT_RTC_ALARM),

                    omap_findclk(s, "clk32-kHz"));



    s->mcbsp1 = omap_mcbsp_init(system_memory, 0xfffb1800,

                                qdev_get_gpio_in(s->ih[1], OMAP_INT_McBSP1TX),

                                qdev_get_gpio_in(s->ih[1], OMAP_INT_McBSP1RX),

                    &s->drq[OMAP_DMA_MCBSP1_TX], omap_findclk(s, "dspxor_ck"));

    s->mcbsp2 = omap_mcbsp_init(system_memory, 0xfffb1000,

                                qdev_get_gpio_in(s->ih[0],

                                                 OMAP_INT_310_McBSP2_TX),

                                qdev_get_gpio_in(s->ih[0],

                                                 OMAP_INT_310_McBSP2_RX),

                    &s->drq[OMAP_DMA_MCBSP2_TX], omap_findclk(s, "mpuper_ck"));

    s->mcbsp3 = omap_mcbsp_init(system_memory, 0xfffb7000,

                                qdev_get_gpio_in(s->ih[1], OMAP_INT_McBSP3TX),

                                qdev_get_gpio_in(s->ih[1], OMAP_INT_McBSP3RX),

                    &s->drq[OMAP_DMA_MCBSP3_TX], omap_findclk(s, "dspxor_ck"));



    s->led[0] = omap_lpg_init(system_memory,

                              0xfffbd000, omap_findclk(s, "clk32-kHz"));

    s->led[1] = omap_lpg_init(system_memory,

                              0xfffbd800, omap_findclk(s, "clk32-kHz"));



    /* Register mappings not currenlty implemented:

     * MCSI2 Comm	fffb2000 - fffb27ff (not mapped on OMAP310)

     * MCSI1 Bluetooth	fffb2800 - fffb2fff (not mapped on OMAP310)

     * USB W2FC		fffb4000 - fffb47ff

     * Camera Interface	fffb6800 - fffb6fff

     * USB Host		fffba000 - fffba7ff

     * FAC		fffba800 - fffbafff

     * HDQ/1-Wire	fffbc000 - fffbc7ff

     * TIPB switches	fffbc800 - fffbcfff

     * Mailbox		fffcf000 - fffcf7ff

     * Local bus IF	fffec100 - fffec1ff

     * Local bus MMU	fffec200 - fffec2ff

     * DSP MMU		fffed200 - fffed2ff

     */



    omap_setup_dsp_mapping(system_memory, omap15xx_dsp_mm);

    omap_setup_mpui_io(system_memory, s);



    qemu_register_reset(omap1_mpu_reset, s);



    return s;

}
