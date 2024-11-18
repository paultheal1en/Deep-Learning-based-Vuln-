static void sb16_realizefn (DeviceState *dev, Error **errp)

{

    ISADevice *isadev = ISA_DEVICE (dev);

    SB16State *s = SB16 (dev);

    IsaDmaClass *k;



    isa_init_irq (isadev, &s->pic, s->irq);



    s->mixer_regs[0x80] = magic_of_irq (s->irq);

    s->mixer_regs[0x81] = (1 << s->dma) | (1 << s->hdma);

    s->mixer_regs[0x82] = 2 << 5;



    s->csp_regs[5] = 1;

    s->csp_regs[9] = 0xf8;



    reset_mixer (s);

    s->aux_ts = timer_new_ns(QEMU_CLOCK_VIRTUAL, aux_timer, s);

    if (!s->aux_ts) {

        dolog ("warning: Could not create auxiliary timer\n");

    }



    isa_register_portio_list (isadev, s->port, sb16_ioport_list, s, "sb16");



    s->isa_hdma = isa_get_dma(isa_bus_from_device(isadev), s->hdma);

    k = ISADMA_GET_CLASS(s->isa_hdma);

    k->register_channel(s->isa_hdma, s->hdma, SB_read_DMA, s);



    s->isa_dma = isa_get_dma(isa_bus_from_device(isadev), s->dma);

    k = ISADMA_GET_CLASS(s->isa_dma);

    k->register_channel(s->isa_dma, s->dma, SB_read_DMA, s);



    s->can_write = 1;



    AUD_register_card ("sb16", &s->card);

}
