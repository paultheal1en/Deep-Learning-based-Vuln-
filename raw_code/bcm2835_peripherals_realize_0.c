static void bcm2835_peripherals_realize(DeviceState *dev, Error **errp)

{

    BCM2835PeripheralState *s = BCM2835_PERIPHERALS(dev);

    Object *obj;

    MemoryRegion *ram;

    Error *err = NULL;

    uint32_t ram_size, vcram_size;

    int n;



    obj = object_property_get_link(OBJECT(dev), "ram", &err);

    if (obj == NULL) {

        error_setg(errp, "%s: required ram link not found: %s",

                   __func__, error_get_pretty(err));

        return;

    }



    ram = MEMORY_REGION(obj);

    ram_size = memory_region_size(ram);



    /* Map peripherals and RAM into the GPU address space. */

    memory_region_init_alias(&s->peri_mr_alias, OBJECT(s),

                             "bcm2835-peripherals", &s->peri_mr, 0,

                             memory_region_size(&s->peri_mr));



    memory_region_add_subregion_overlap(&s->gpu_bus_mr, BCM2835_VC_PERI_BASE,

                                        &s->peri_mr_alias, 1);



    /* RAM is aliased four times (different cache configurations) on the GPU */

    for (n = 0; n < 4; n++) {

        memory_region_init_alias(&s->ram_alias[n], OBJECT(s),

                                 "bcm2835-gpu-ram-alias[*]", ram, 0, ram_size);

        memory_region_add_subregion_overlap(&s->gpu_bus_mr, (hwaddr)n << 30,

                                            &s->ram_alias[n], 0);

    }



    /* Interrupt Controller */

    object_property_set_bool(OBJECT(&s->ic), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    memory_region_add_subregion(&s->peri_mr, ARMCTRL_IC_OFFSET,

                sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->ic), 0));

    sysbus_pass_irq(SYS_BUS_DEVICE(s), SYS_BUS_DEVICE(&s->ic));



    /* UART0 */

    qdev_prop_set_chr(DEVICE(s->uart0), "chardev", serial_hds[0]);

    object_property_set_bool(OBJECT(s->uart0), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    memory_region_add_subregion(&s->peri_mr, UART0_OFFSET,

                                sysbus_mmio_get_region(s->uart0, 0));

    sysbus_connect_irq(s->uart0, 0,

        qdev_get_gpio_in_named(DEVICE(&s->ic), BCM2835_IC_GPU_IRQ,

                               INTERRUPT_UART));

    /* AUX / UART1 */

    qdev_prop_set_chr(DEVICE(&s->aux), "chardev", serial_hds[1]);



    object_property_set_bool(OBJECT(&s->aux), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    memory_region_add_subregion(&s->peri_mr, UART1_OFFSET,

                sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->aux), 0));

    sysbus_connect_irq(SYS_BUS_DEVICE(&s->aux), 0,

        qdev_get_gpio_in_named(DEVICE(&s->ic), BCM2835_IC_GPU_IRQ,

                               INTERRUPT_AUX));



    /* Mailboxes */

    object_property_set_bool(OBJECT(&s->mboxes), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    memory_region_add_subregion(&s->peri_mr, ARMCTRL_0_SBM_OFFSET,

                sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->mboxes), 0));

    sysbus_connect_irq(SYS_BUS_DEVICE(&s->mboxes), 0,

        qdev_get_gpio_in_named(DEVICE(&s->ic), BCM2835_IC_ARM_IRQ,

                               INTERRUPT_ARM_MAILBOX));



    /* Framebuffer */

    vcram_size = (uint32_t)object_property_get_int(OBJECT(s), "vcram-size",

                                                   &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    object_property_set_int(OBJECT(&s->fb), ram_size - vcram_size,

                            "vcram-base", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    object_property_set_bool(OBJECT(&s->fb), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    memory_region_add_subregion(&s->mbox_mr, MBOX_CHAN_FB << MBOX_AS_CHAN_SHIFT,

                sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->fb), 0));

    sysbus_connect_irq(SYS_BUS_DEVICE(&s->fb), 0,

                       qdev_get_gpio_in(DEVICE(&s->mboxes), MBOX_CHAN_FB));



    /* Property channel */

    object_property_set_bool(OBJECT(&s->property), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    memory_region_add_subregion(&s->mbox_mr,

                MBOX_CHAN_PROPERTY << MBOX_AS_CHAN_SHIFT,

                sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->property), 0));

    sysbus_connect_irq(SYS_BUS_DEVICE(&s->property), 0,

                      qdev_get_gpio_in(DEVICE(&s->mboxes), MBOX_CHAN_PROPERTY));



    /* Random Number Generator */

    object_property_set_bool(OBJECT(&s->rng), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    memory_region_add_subregion(&s->peri_mr, RNG_OFFSET,

                sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->rng), 0));



    /* Extended Mass Media Controller */

    object_property_set_int(OBJECT(&s->sdhci), BCM2835_SDHC_CAPAREG, "capareg",

                            &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    object_property_set_bool(OBJECT(&s->sdhci), true, "pending-insert-quirk",

                             &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    object_property_set_bool(OBJECT(&s->sdhci), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    memory_region_add_subregion(&s->peri_mr, EMMC_OFFSET,

                sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->sdhci), 0));

    sysbus_connect_irq(SYS_BUS_DEVICE(&s->sdhci), 0,

        qdev_get_gpio_in_named(DEVICE(&s->ic), BCM2835_IC_GPU_IRQ,

                               INTERRUPT_ARASANSDIO));

    object_property_add_alias(OBJECT(s), "sd-bus", OBJECT(&s->sdhci), "sd-bus",

                              &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    /* DMA Channels */

    object_property_set_bool(OBJECT(&s->dma), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    memory_region_add_subregion(&s->peri_mr, DMA_OFFSET,

                sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->dma), 0));

    memory_region_add_subregion(&s->peri_mr, DMA15_OFFSET,

                sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->dma), 1));



    for (n = 0; n <= 12; n++) {

        sysbus_connect_irq(SYS_BUS_DEVICE(&s->dma), n,

                           qdev_get_gpio_in_named(DEVICE(&s->ic),

                                                  BCM2835_IC_GPU_IRQ,

                                                  INTERRUPT_DMA0 + n));

    }

}
