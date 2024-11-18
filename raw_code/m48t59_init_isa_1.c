m48t59_t *m48t59_init_isa(uint32_t io_base, uint16_t size, int type)

{

    M48t59ISAState *d;

    ISADevice *dev;

    m48t59_t *s;



    dev = isa_create("m48t59_isa");

    qdev_prop_set_uint32(&dev->qdev, "type", type);

    qdev_prop_set_uint32(&dev->qdev, "size", size);

    qdev_prop_set_uint32(&dev->qdev, "io_base", io_base);

    qdev_init(&dev->qdev);

    d = DO_UPCAST(M48t59ISAState, busdev, dev);

    s = &d->state;



    if (io_base != 0) {

        register_ioport_read(io_base, 0x04, 1, NVRAM_readb, s);

        register_ioport_write(io_base, 0x04, 1, NVRAM_writeb, s);

    }



    return s;

}
