Exynos4210State *exynos4210_init(MemoryRegion *system_mem,

        unsigned long ram_size)

{

    int i, n;

    Exynos4210State *s = g_new(Exynos4210State, 1);

    qemu_irq gate_irq[EXYNOS4210_NCPUS][EXYNOS4210_IRQ_GATE_NINPUTS];

    unsigned long mem_size;

    DeviceState *dev;

    SysBusDevice *busdev;

    ObjectClass *cpu_oc;



    cpu_oc = cpu_class_by_name(TYPE_ARM_CPU, "cortex-a9");

    assert(cpu_oc);



    for (n = 0; n < EXYNOS4210_NCPUS; n++) {

        Object *cpuobj = object_new(object_class_get_name(cpu_oc));

        Error *err = NULL;



        /* By default A9 CPUs have EL3 enabled.  This board does not currently

         * support EL3 so the CPU EL3 property is disabled before realization.

         */

        if (object_property_find(cpuobj, "has_el3", NULL)) {

            object_property_set_bool(cpuobj, false, "has_el3", &err);

            if (err) {

                error_report_err(err);

                exit(1);

            }

        }



        s->cpu[n] = ARM_CPU(cpuobj);

        object_property_set_int(cpuobj, EXYNOS4210_SMP_PRIVATE_BASE_ADDR,

                                "reset-cbar", &error_abort);

        object_property_set_bool(cpuobj, true, "realized", &err);

        if (err) {

            error_report_err(err);

            exit(1);

        }

    }



    /*** IRQs ***/



    s->irq_table = exynos4210_init_irq(&s->irqs);



    /* IRQ Gate */

    for (i = 0; i < EXYNOS4210_NCPUS; i++) {

        dev = qdev_create(NULL, "exynos4210.irq_gate");

        qdev_prop_set_uint32(dev, "n_in", EXYNOS4210_IRQ_GATE_NINPUTS);

        qdev_init_nofail(dev);

        /* Get IRQ Gate input in gate_irq */

        for (n = 0; n < EXYNOS4210_IRQ_GATE_NINPUTS; n++) {

            gate_irq[i][n] = qdev_get_gpio_in(dev, n);

        }

        busdev = SYS_BUS_DEVICE(dev);



        /* Connect IRQ Gate output to CPU's IRQ line */

        sysbus_connect_irq(busdev, 0,

                           qdev_get_gpio_in(DEVICE(s->cpu[i]), ARM_CPU_IRQ));

    }



    /* Private memory region and Internal GIC */

    dev = qdev_create(NULL, "a9mpcore_priv");

    qdev_prop_set_uint32(dev, "num-cpu", EXYNOS4210_NCPUS);

    qdev_init_nofail(dev);

    busdev = SYS_BUS_DEVICE(dev);

    sysbus_mmio_map(busdev, 0, EXYNOS4210_SMP_PRIVATE_BASE_ADDR);

    for (n = 0; n < EXYNOS4210_NCPUS; n++) {

        sysbus_connect_irq(busdev, n, gate_irq[n][0]);

    }

    for (n = 0; n < EXYNOS4210_INT_GIC_NIRQ; n++) {

        s->irqs.int_gic_irq[n] = qdev_get_gpio_in(dev, n);

    }



    /* Cache controller */

    sysbus_create_simple("l2x0", EXYNOS4210_L2X0_BASE_ADDR, NULL);



    /* External GIC */

    dev = qdev_create(NULL, "exynos4210.gic");

    qdev_prop_set_uint32(dev, "num-cpu", EXYNOS4210_NCPUS);

    qdev_init_nofail(dev);

    busdev = SYS_BUS_DEVICE(dev);

    /* Map CPU interface */

    sysbus_mmio_map(busdev, 0, EXYNOS4210_EXT_GIC_CPU_BASE_ADDR);

    /* Map Distributer interface */

    sysbus_mmio_map(busdev, 1, EXYNOS4210_EXT_GIC_DIST_BASE_ADDR);

    for (n = 0; n < EXYNOS4210_NCPUS; n++) {

        sysbus_connect_irq(busdev, n, gate_irq[n][1]);

    }

    for (n = 0; n < EXYNOS4210_EXT_GIC_NIRQ; n++) {

        s->irqs.ext_gic_irq[n] = qdev_get_gpio_in(dev, n);

    }



    /* Internal Interrupt Combiner */

    dev = qdev_create(NULL, "exynos4210.combiner");

    qdev_init_nofail(dev);

    busdev = SYS_BUS_DEVICE(dev);

    for (n = 0; n < EXYNOS4210_MAX_INT_COMBINER_OUT_IRQ; n++) {

        sysbus_connect_irq(busdev, n, s->irqs.int_gic_irq[n]);

    }

    exynos4210_combiner_get_gpioin(&s->irqs, dev, 0);

    sysbus_mmio_map(busdev, 0, EXYNOS4210_INT_COMBINER_BASE_ADDR);



    /* External Interrupt Combiner */

    dev = qdev_create(NULL, "exynos4210.combiner");

    qdev_prop_set_uint32(dev, "external", 1);

    qdev_init_nofail(dev);

    busdev = SYS_BUS_DEVICE(dev);

    for (n = 0; n < EXYNOS4210_MAX_INT_COMBINER_OUT_IRQ; n++) {

        sysbus_connect_irq(busdev, n, s->irqs.ext_gic_irq[n]);

    }

    exynos4210_combiner_get_gpioin(&s->irqs, dev, 1);

    sysbus_mmio_map(busdev, 0, EXYNOS4210_EXT_COMBINER_BASE_ADDR);



    /* Initialize board IRQs. */

    exynos4210_init_board_irqs(&s->irqs);



    /*** Memory ***/



    /* Chip-ID and OMR */

    memory_region_init_io(&s->chipid_mem, NULL, &exynos4210_chipid_and_omr_ops,

        NULL, "exynos4210.chipid", sizeof(chipid_and_omr));

    memory_region_add_subregion(system_mem, EXYNOS4210_CHIPID_ADDR,

                                &s->chipid_mem);



    /* Internal ROM */

    memory_region_init_ram(&s->irom_mem, NULL, "exynos4210.irom",

                           EXYNOS4210_IROM_SIZE, &error_abort);

    vmstate_register_ram_global(&s->irom_mem);

    memory_region_set_readonly(&s->irom_mem, true);

    memory_region_add_subregion(system_mem, EXYNOS4210_IROM_BASE_ADDR,

                                &s->irom_mem);

    /* mirror of iROM */

    memory_region_init_alias(&s->irom_alias_mem, NULL, "exynos4210.irom_alias",

                             &s->irom_mem,

                             0,

                             EXYNOS4210_IROM_SIZE);

    memory_region_set_readonly(&s->irom_alias_mem, true);

    memory_region_add_subregion(system_mem, EXYNOS4210_IROM_MIRROR_BASE_ADDR,

                                &s->irom_alias_mem);



    /* Internal RAM */

    memory_region_init_ram(&s->iram_mem, NULL, "exynos4210.iram",

                           EXYNOS4210_IRAM_SIZE, &error_abort);

    vmstate_register_ram_global(&s->iram_mem);

    memory_region_add_subregion(system_mem, EXYNOS4210_IRAM_BASE_ADDR,

                                &s->iram_mem);



    /* DRAM */

    mem_size = ram_size;

    if (mem_size > EXYNOS4210_DRAM_MAX_SIZE) {

        memory_region_init_ram(&s->dram1_mem, NULL, "exynos4210.dram1",

                mem_size - EXYNOS4210_DRAM_MAX_SIZE, &error_abort);

        vmstate_register_ram_global(&s->dram1_mem);

        memory_region_add_subregion(system_mem, EXYNOS4210_DRAM1_BASE_ADDR,

                &s->dram1_mem);

        mem_size = EXYNOS4210_DRAM_MAX_SIZE;

    }

    memory_region_init_ram(&s->dram0_mem, NULL, "exynos4210.dram0", mem_size,

                           &error_abort);

    vmstate_register_ram_global(&s->dram0_mem);

    memory_region_add_subregion(system_mem, EXYNOS4210_DRAM0_BASE_ADDR,

            &s->dram0_mem);



   /* PMU.

    * The only reason of existence at the moment is that secondary CPU boot

    * loader uses PMU INFORM5 register as a holding pen.

    */

    sysbus_create_simple("exynos4210.pmu", EXYNOS4210_PMU_BASE_ADDR, NULL);



    /* PWM */

    sysbus_create_varargs("exynos4210.pwm", EXYNOS4210_PWM_BASE_ADDR,

                          s->irq_table[exynos4210_get_irq(22, 0)],

                          s->irq_table[exynos4210_get_irq(22, 1)],

                          s->irq_table[exynos4210_get_irq(22, 2)],

                          s->irq_table[exynos4210_get_irq(22, 3)],

                          s->irq_table[exynos4210_get_irq(22, 4)],

                          NULL);

    /* RTC */

    sysbus_create_varargs("exynos4210.rtc", EXYNOS4210_RTC_BASE_ADDR,

                          s->irq_table[exynos4210_get_irq(23, 0)],

                          s->irq_table[exynos4210_get_irq(23, 1)],

                          NULL);



    /* Multi Core Timer */

    dev = qdev_create(NULL, "exynos4210.mct");

    qdev_init_nofail(dev);

    busdev = SYS_BUS_DEVICE(dev);

    for (n = 0; n < 4; n++) {

        /* Connect global timer interrupts to Combiner gpio_in */

        sysbus_connect_irq(busdev, n,

                s->irq_table[exynos4210_get_irq(1, 4 + n)]);

    }

    /* Connect local timer interrupts to Combiner gpio_in */

    sysbus_connect_irq(busdev, 4,

            s->irq_table[exynos4210_get_irq(51, 0)]);

    sysbus_connect_irq(busdev, 5,

            s->irq_table[exynos4210_get_irq(35, 3)]);

    sysbus_mmio_map(busdev, 0, EXYNOS4210_MCT_BASE_ADDR);



    /*** I2C ***/

    for (n = 0; n < EXYNOS4210_I2C_NUMBER; n++) {

        uint32_t addr = EXYNOS4210_I2C_BASE_ADDR + EXYNOS4210_I2C_SHIFT * n;

        qemu_irq i2c_irq;



        if (n < 8) {

            i2c_irq = s->irq_table[exynos4210_get_irq(EXYNOS4210_I2C_INTG, n)];

        } else {

            i2c_irq = s->irq_table[exynos4210_get_irq(EXYNOS4210_HDMI_INTG, 1)];

        }



        dev = qdev_create(NULL, "exynos4210.i2c");

        qdev_init_nofail(dev);

        busdev = SYS_BUS_DEVICE(dev);

        sysbus_connect_irq(busdev, 0, i2c_irq);

        sysbus_mmio_map(busdev, 0, addr);

        s->i2c_if[n] = (I2CBus *)qdev_get_child_bus(dev, "i2c");

    }





    /*** UARTs ***/

    exynos4210_uart_create(EXYNOS4210_UART0_BASE_ADDR,

                           EXYNOS4210_UART0_FIFO_SIZE, 0, NULL,

                  s->irq_table[exynos4210_get_irq(EXYNOS4210_UART_INT_GRP, 0)]);



    exynos4210_uart_create(EXYNOS4210_UART1_BASE_ADDR,

                           EXYNOS4210_UART1_FIFO_SIZE, 1, NULL,

                  s->irq_table[exynos4210_get_irq(EXYNOS4210_UART_INT_GRP, 1)]);



    exynos4210_uart_create(EXYNOS4210_UART2_BASE_ADDR,

                           EXYNOS4210_UART2_FIFO_SIZE, 2, NULL,

                  s->irq_table[exynos4210_get_irq(EXYNOS4210_UART_INT_GRP, 2)]);



    exynos4210_uart_create(EXYNOS4210_UART3_BASE_ADDR,

                           EXYNOS4210_UART3_FIFO_SIZE, 3, NULL,

                  s->irq_table[exynos4210_get_irq(EXYNOS4210_UART_INT_GRP, 3)]);



    /*** Display controller (FIMD) ***/

    sysbus_create_varargs("exynos4210.fimd", EXYNOS4210_FIMD0_BASE_ADDR,

            s->irq_table[exynos4210_get_irq(11, 0)],

            s->irq_table[exynos4210_get_irq(11, 1)],

            s->irq_table[exynos4210_get_irq(11, 2)],

            NULL);



    sysbus_create_simple(TYPE_EXYNOS4210_EHCI, EXYNOS4210_EHCI_BASE_ADDR,

            s->irq_table[exynos4210_get_irq(28, 3)]);



    return s;

}
