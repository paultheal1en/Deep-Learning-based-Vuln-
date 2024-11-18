static void i8257_realize(DeviceState *dev, Error **errp)

{

    ISADevice *isa = ISA_DEVICE(dev);

    I8257State *d = I8257(dev);

    int i;



    memory_region_init_io(&d->channel_io, NULL, &channel_io_ops, d,

                          "dma-chan", 8 << d->dshift);

    memory_region_add_subregion(isa_address_space_io(isa),

                                d->base, &d->channel_io);



    isa_register_portio_list(isa, d->page_base, page_portio_list, d,

                             "dma-page");

    if (d->pageh_base >= 0) {

        isa_register_portio_list(isa, d->pageh_base, pageh_portio_list, d,

                                 "dma-pageh");

    }



    memory_region_init_io(&d->cont_io, OBJECT(isa), &cont_io_ops, d,

                          "dma-cont", 8 << d->dshift);

    memory_region_add_subregion(isa_address_space_io(isa),

                                d->base + (8 << d->dshift), &d->cont_io);



    for (i = 0; i < ARRAY_SIZE(d->regs); ++i) {

        d->regs[i].transfer_handler = i8257_phony_handler;

    }



    d->dma_bh = qemu_bh_new(i8257_dma_run, d);

}
