static int integratorcm_init(SysBusDevice *dev)

{

    IntegratorCMState *s = INTEGRATOR_CM(dev);



    s->cm_osc = 0x01000048;

    /* ??? What should the high bits of this value be?  */

    s->cm_auxosc = 0x0007feff;

    s->cm_sdram = 0x00011122;

    if (s->memsz >= 256) {

        integrator_spd[31] = 64;

        s->cm_sdram |= 0x10;

    } else if (s->memsz >= 128) {

        integrator_spd[31] = 32;

        s->cm_sdram |= 0x0c;

    } else if (s->memsz >= 64) {

        integrator_spd[31] = 16;

        s->cm_sdram |= 0x08;

    } else if (s->memsz >= 32) {

        integrator_spd[31] = 4;

        s->cm_sdram |= 0x04;

    } else {

        integrator_spd[31] = 2;

    }

    memcpy(integrator_spd + 73, "QEMU-MEMORY", 11);

    s->cm_init = 0x00000112;

    s->cm_refcnt_offset = muldiv64(qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL), 24,

                                   1000);

    memory_region_init_ram(&s->flash, OBJECT(s), "integrator.flash", 0x100000,

                           &error_abort);

    vmstate_register_ram_global(&s->flash);



    memory_region_init_io(&s->iomem, OBJECT(s), &integratorcm_ops, s,

                          "integratorcm", 0x00800000);

    sysbus_init_mmio(dev, &s->iomem);



    integratorcm_do_remap(s);

    /* ??? Save/restore.  */

    return 0;

}
