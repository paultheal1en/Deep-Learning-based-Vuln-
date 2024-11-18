static int pxa2xx_timer_init(SysBusDevice *dev)

{

    int i;

    int iomemtype;

    PXA2xxTimerInfo *s;

    qemu_irq irq4;



    s = FROM_SYSBUS(PXA2xxTimerInfo, dev);

    s->irq_enabled = 0;

    s->oldclock = 0;

    s->clock = 0;

    s->lastload = qemu_get_clock(vm_clock);

    s->reset3 = 0;



    for (i = 0; i < 4; i ++) {

        s->timer[i].value = 0;

        sysbus_init_irq(dev, &s->timer[i].irq);

        s->timer[i].info = s;

        s->timer[i].num = i;

        s->timer[i].level = 0;

        s->timer[i].qtimer = qemu_new_timer(vm_clock,

                        pxa2xx_timer_tick, &s->timer[i]);

    }

    if (s->flags & (1 << PXA2XX_TIMER_HAVE_TM4)) {

        sysbus_init_irq(dev, &irq4);



        for (i = 0; i < 8; i ++) {

            s->tm4[i].tm.value = 0;

            s->tm4[i].tm.info = s;

            s->tm4[i].tm.num = i + 4;

            s->tm4[i].tm.level = 0;

            s->tm4[i].freq = 0;

            s->tm4[i].control = 0x0;

            s->tm4[i].tm.qtimer = qemu_new_timer(vm_clock,

                        pxa2xx_timer_tick4, &s->tm4[i]);

            s->tm4[i].tm.irq = irq4;

        }

    }



    iomemtype = cpu_register_io_memory(pxa2xx_timer_readfn,

                    pxa2xx_timer_writefn, s, DEVICE_NATIVE_ENDIAN);

    sysbus_init_mmio(dev, 0x00001000, iomemtype);



    return 0;

}
