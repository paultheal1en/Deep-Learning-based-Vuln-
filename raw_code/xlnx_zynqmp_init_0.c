static void xlnx_zynqmp_init(Object *obj)

{

    XlnxZynqMPState *s = XLNX_ZYNQMP(obj);

    int i;



    for (i = 0; i < XLNX_ZYNQMP_NUM_APU_CPUS; i++) {

        object_initialize(&s->apu_cpu[i], sizeof(s->apu_cpu[i]),

                          "cortex-a53-" TYPE_ARM_CPU);

        object_property_add_child(obj, "apu-cpu[*]", OBJECT(&s->apu_cpu[i]),

                                  &error_abort);

    }



    for (i = 0; i < XLNX_ZYNQMP_NUM_RPU_CPUS; i++) {

        object_initialize(&s->rpu_cpu[i], sizeof(s->rpu_cpu[i]),

                          "cortex-r5-" TYPE_ARM_CPU);

        object_property_add_child(obj, "rpu-cpu[*]", OBJECT(&s->rpu_cpu[i]),

                                  &error_abort);

    }



    object_property_add_link(obj, "ddr-ram", TYPE_MEMORY_REGION,

                             (Object **)&s->ddr_ram,

                             qdev_prop_allow_set_link_before_realize,

                             OBJ_PROP_LINK_UNREF_ON_RELEASE, &error_abort);



    object_initialize(&s->gic, sizeof(s->gic), TYPE_ARM_GIC);

    qdev_set_parent_bus(DEVICE(&s->gic), sysbus_get_default());



    for (i = 0; i < XLNX_ZYNQMP_NUM_GEMS; i++) {

        object_initialize(&s->gem[i], sizeof(s->gem[i]), TYPE_CADENCE_GEM);

        qdev_set_parent_bus(DEVICE(&s->gem[i]), sysbus_get_default());

    }



    for (i = 0; i < XLNX_ZYNQMP_NUM_UARTS; i++) {

        object_initialize(&s->uart[i], sizeof(s->uart[i]), TYPE_CADENCE_UART);

        qdev_set_parent_bus(DEVICE(&s->uart[i]), sysbus_get_default());

    }



    object_initialize(&s->sata, sizeof(s->sata), TYPE_SYSBUS_AHCI);

    qdev_set_parent_bus(DEVICE(&s->sata), sysbus_get_default());



    for (i = 0; i < XLNX_ZYNQMP_NUM_SDHCI; i++) {

        object_initialize(&s->sdhci[i], sizeof(s->sdhci[i]),

                          TYPE_SYSBUS_SDHCI);

        qdev_set_parent_bus(DEVICE(&s->sdhci[i]),

                            sysbus_get_default());

    }



    for (i = 0; i < XLNX_ZYNQMP_NUM_SPIS; i++) {

        object_initialize(&s->spi[i], sizeof(s->spi[i]),

                          TYPE_XILINX_SPIPS);

        qdev_set_parent_bus(DEVICE(&s->spi[i]), sysbus_get_default());

    }

}
