static void xlnx_zynqmp_realize(DeviceState *dev, Error **errp)

{

    XlnxZynqMPState *s = XLNX_ZYNQMP(dev);

    MemoryRegion *system_memory = get_system_memory();

    uint8_t i;

    const char *boot_cpu = s->boot_cpu ? s->boot_cpu : "apu-cpu[0]";

    qemu_irq gic_spi[GIC_NUM_SPI_INTR];

    Error *err = NULL;



    /* Create the four OCM banks */

    for (i = 0; i < XLNX_ZYNQMP_NUM_OCM_BANKS; i++) {

        char *ocm_name = g_strdup_printf("zynqmp.ocm_ram_bank_%d", i);



        memory_region_init_ram(&s->ocm_ram[i], NULL, ocm_name,

                               XLNX_ZYNQMP_OCM_RAM_SIZE, &error_abort);

        vmstate_register_ram_global(&s->ocm_ram[i]);

        memory_region_add_subregion(get_system_memory(),

                                    XLNX_ZYNQMP_OCM_RAM_0_ADDRESS +

                                        i * XLNX_ZYNQMP_OCM_RAM_SIZE,

                                    &s->ocm_ram[i]);



        g_free(ocm_name);

    }



    qdev_prop_set_uint32(DEVICE(&s->gic), "num-irq", GIC_NUM_SPI_INTR + 32);

    qdev_prop_set_uint32(DEVICE(&s->gic), "revision", 2);

    qdev_prop_set_uint32(DEVICE(&s->gic), "num-cpu", XLNX_ZYNQMP_NUM_APU_CPUS);

    object_property_set_bool(OBJECT(&s->gic), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    assert(ARRAY_SIZE(xlnx_zynqmp_gic_regions) == XLNX_ZYNQMP_GIC_REGIONS);

    for (i = 0; i < XLNX_ZYNQMP_GIC_REGIONS; i++) {

        SysBusDevice *gic = SYS_BUS_DEVICE(&s->gic);

        const XlnxZynqMPGICRegion *r = &xlnx_zynqmp_gic_regions[i];

        MemoryRegion *mr = sysbus_mmio_get_region(gic, r->region_index);

        uint32_t addr = r->address;

        int j;



        sysbus_mmio_map(gic, r->region_index, addr);



        for (j = 0; j < XLNX_ZYNQMP_GIC_ALIASES; j++) {

            MemoryRegion *alias = &s->gic_mr[i][j];



            addr += XLNX_ZYNQMP_GIC_REGION_SIZE;

            memory_region_init_alias(alias, OBJECT(s), "zynqmp-gic-alias", mr,

                                     0, XLNX_ZYNQMP_GIC_REGION_SIZE);

            memory_region_add_subregion(system_memory, addr, alias);

        }

    }



    for (i = 0; i < XLNX_ZYNQMP_NUM_APU_CPUS; i++) {

        qemu_irq irq;

        char *name;



        object_property_set_int(OBJECT(&s->apu_cpu[i]), QEMU_PSCI_CONDUIT_SMC,

                                "psci-conduit", &error_abort);



        name = object_get_canonical_path_component(OBJECT(&s->apu_cpu[i]));

        if (strcmp(name, boot_cpu)) {

            /* Secondary CPUs start in PSCI powered-down state */

            object_property_set_bool(OBJECT(&s->apu_cpu[i]), true,

                                     "start-powered-off", &error_abort);

        } else {

            s->boot_cpu_ptr = &s->apu_cpu[i];

        }

        g_free(name);



        object_property_set_int(OBJECT(&s->apu_cpu[i]), GIC_BASE_ADDR,

                                "reset-cbar", &error_abort);

        object_property_set_bool(OBJECT(&s->apu_cpu[i]), true, "realized",

                                 &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }



        sysbus_connect_irq(SYS_BUS_DEVICE(&s->gic), i,

                           qdev_get_gpio_in(DEVICE(&s->apu_cpu[i]),

                                            ARM_CPU_IRQ));

        irq = qdev_get_gpio_in(DEVICE(&s->gic),

                               arm_gic_ppi_index(i, ARM_PHYS_TIMER_PPI));

        qdev_connect_gpio_out(DEVICE(&s->apu_cpu[i]), 0, irq);

        irq = qdev_get_gpio_in(DEVICE(&s->gic),

                               arm_gic_ppi_index(i, ARM_VIRT_TIMER_PPI));

        qdev_connect_gpio_out(DEVICE(&s->apu_cpu[i]), 1, irq);

    }



    for (i = 0; i < XLNX_ZYNQMP_NUM_RPU_CPUS; i++) {

        char *name;



        name = object_get_canonical_path_component(OBJECT(&s->rpu_cpu[i]));

        if (strcmp(name, boot_cpu)) {

            /* Secondary CPUs start in PSCI powered-down state */

            object_property_set_bool(OBJECT(&s->rpu_cpu[i]), true,

                                     "start-powered-off", &error_abort);

        } else {

            s->boot_cpu_ptr = &s->rpu_cpu[i];

        }

        g_free(name);



        object_property_set_bool(OBJECT(&s->rpu_cpu[i]), true, "reset-hivecs",

                                 &error_abort);

        object_property_set_bool(OBJECT(&s->rpu_cpu[i]), true, "realized",

                                 &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }

    }



    if (!s->boot_cpu_ptr) {

        error_setg(errp, "ZynqMP Boot cpu %s not found\n", boot_cpu);

        return;

    }



    for (i = 0; i < GIC_NUM_SPI_INTR; i++) {

        gic_spi[i] = qdev_get_gpio_in(DEVICE(&s->gic), i);

    }



    for (i = 0; i < XLNX_ZYNQMP_NUM_GEMS; i++) {

        NICInfo *nd = &nd_table[i];



        if (nd->used) {

            qemu_check_nic_model(nd, TYPE_CADENCE_GEM);

            qdev_set_nic_properties(DEVICE(&s->gem[i]), nd);

        }

        object_property_set_bool(OBJECT(&s->gem[i]), true, "realized", &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }

        sysbus_mmio_map(SYS_BUS_DEVICE(&s->gem[i]), 0, gem_addr[i]);

        sysbus_connect_irq(SYS_BUS_DEVICE(&s->gem[i]), 0,

                           gic_spi[gem_intr[i]]);

    }



    for (i = 0; i < XLNX_ZYNQMP_NUM_UARTS; i++) {

        object_property_set_bool(OBJECT(&s->uart[i]), true, "realized", &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }

        sysbus_mmio_map(SYS_BUS_DEVICE(&s->uart[i]), 0, uart_addr[i]);

        sysbus_connect_irq(SYS_BUS_DEVICE(&s->uart[i]), 0,

                           gic_spi[uart_intr[i]]);

    }



    object_property_set_int(OBJECT(&s->sata), SATA_NUM_PORTS, "num-ports",

                            &error_abort);

    object_property_set_bool(OBJECT(&s->sata), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    sysbus_mmio_map(SYS_BUS_DEVICE(&s->sata), 0, SATA_ADDR);

    sysbus_connect_irq(SYS_BUS_DEVICE(&s->sata), 0, gic_spi[SATA_INTR]);

}
