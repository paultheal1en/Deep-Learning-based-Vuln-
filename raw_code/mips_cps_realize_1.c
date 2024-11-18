static void mips_cps_realize(DeviceState *dev, Error **errp)

{

    MIPSCPSState *s = MIPS_CPS(dev);

    CPUMIPSState *env;

    MIPSCPU *cpu;

    int i;

    Error *err = NULL;

    target_ulong gcr_base;

    bool itu_present = false;



    for (i = 0; i < s->num_vp; i++) {

        cpu = cpu_mips_init(s->cpu_model);

        if (cpu == NULL) {

            error_setg(errp, "%s: CPU initialization failed",  __func__);

            return;

        }



        /* Init internal devices */

        cpu_mips_irq_init_cpu(cpu);

        cpu_mips_clock_init(cpu);



        env = &cpu->env;

        if (cpu_mips_itu_supported(env)) {

            itu_present = true;

            /* Attach ITC Tag to the VP */

            env->itc_tag = mips_itu_get_tag_region(&s->itu);

        }

        qemu_register_reset(main_cpu_reset, cpu);

    }



    cpu = MIPS_CPU(first_cpu);

    env = &cpu->env;



    /* Inter-Thread Communication Unit */

    if (itu_present) {

        object_initialize(&s->itu, sizeof(s->itu), TYPE_MIPS_ITU);

        qdev_set_parent_bus(DEVICE(&s->itu), sysbus_get_default());



        object_property_set_int(OBJECT(&s->itu), 16, "num-fifo", &err);

        object_property_set_int(OBJECT(&s->itu), 16, "num-semaphores", &err);

        object_property_set_bool(OBJECT(&s->itu), true, "realized", &err);

        if (err != NULL) {

            error_propagate(errp, err);

            return;

        }



        memory_region_add_subregion(&s->container, 0,

                           sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->itu), 0));

    }



    /* Cluster Power Controller */

    object_initialize(&s->cpc, sizeof(s->cpc), TYPE_MIPS_CPC);

    qdev_set_parent_bus(DEVICE(&s->cpc), sysbus_get_default());



    object_property_set_int(OBJECT(&s->cpc), s->num_vp, "num-vp", &err);

    object_property_set_int(OBJECT(&s->cpc), 1, "vp-start-running", &err);

    object_property_set_bool(OBJECT(&s->cpc), true, "realized", &err);

    if (err != NULL) {

        error_propagate(errp, err);

        return;

    }



    memory_region_add_subregion(&s->container, 0,

                            sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->cpc), 0));



    /* Global Interrupt Controller */

    object_initialize(&s->gic, sizeof(s->gic), TYPE_MIPS_GIC);

    qdev_set_parent_bus(DEVICE(&s->gic), sysbus_get_default());



    object_property_set_int(OBJECT(&s->gic), s->num_vp, "num-vp", &err);

    object_property_set_int(OBJECT(&s->gic), 128, "num-irq", &err);

    object_property_set_bool(OBJECT(&s->gic), true, "realized", &err);

    if (err != NULL) {

        error_propagate(errp, err);

        return;

    }



    memory_region_add_subregion(&s->container, 0,

                            sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->gic), 0));



    /* Global Configuration Registers */

    gcr_base = env->CP0_CMGCRBase << 4;



    object_initialize(&s->gcr, sizeof(s->gcr), TYPE_MIPS_GCR);

    qdev_set_parent_bus(DEVICE(&s->gcr), sysbus_get_default());



    object_property_set_int(OBJECT(&s->gcr), s->num_vp, "num-vp", &err);

    object_property_set_int(OBJECT(&s->gcr), 0x800, "gcr-rev", &err);

    object_property_set_int(OBJECT(&s->gcr), gcr_base, "gcr-base", &err);

    object_property_set_link(OBJECT(&s->gcr), OBJECT(&s->gic.mr), "gic", &err);

    object_property_set_link(OBJECT(&s->gcr), OBJECT(&s->cpc.mr), "cpc", &err);

    object_property_set_bool(OBJECT(&s->gcr), true, "realized", &err);

    if (err != NULL) {

        error_propagate(errp, err);

        return;

    }



    memory_region_add_subregion(&s->container, gcr_base,

                            sysbus_mmio_get_region(SYS_BUS_DEVICE(&s->gcr), 0));

}
