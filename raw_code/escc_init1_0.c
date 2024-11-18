static int escc_init1(SysBusDevice *dev)

{

    ESCCState *s = ESCC(dev);

    unsigned int i;



    s->chn[0].disabled = s->disabled;

    s->chn[1].disabled = s->disabled;

    for (i = 0; i < 2; i++) {

        sysbus_init_irq(dev, &s->chn[i].irq);

        s->chn[i].chn = 1 - i;

        s->chn[i].clock = s->frequency / 2;

        if (s->chn[i].chr) {

            qemu_chr_add_handlers(s->chn[i].chr, serial_can_receive,

                                  serial_receive1, serial_event, &s->chn[i]);

        }

    }

    s->chn[0].otherchn = &s->chn[1];

    s->chn[1].otherchn = &s->chn[0];



    memory_region_init_io(&s->mmio, OBJECT(s), &escc_mem_ops, s, "escc",

                          ESCC_SIZE << s->it_shift);

    sysbus_init_mmio(dev, &s->mmio);



    if (s->chn[0].type == mouse) {

        qemu_add_mouse_event_handler(sunmouse_event, &s->chn[0], 0,

                                     "QEMU Sun Mouse");

    }

    if (s->chn[1].type == kbd) {

        qemu_add_kbd_event_handler(sunkbd_event, &s->chn[1]);

    }



    return 0;

}
