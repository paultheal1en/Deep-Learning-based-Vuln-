static int pl061_init(SysBusDevice *dev, const unsigned char *id)

{

    int iomemtype;

    pl061_state *s = FROM_SYSBUS(pl061_state, dev);

    s->id = id;

    iomemtype = cpu_register_io_memory(pl061_readfn,

                                       pl061_writefn, s,

                                       DEVICE_NATIVE_ENDIAN);

    sysbus_init_mmio(dev, 0x1000, iomemtype);

    sysbus_init_irq(dev, &s->irq);

    qdev_init_gpio_in(&dev->qdev, pl061_set_irq, 8);

    qdev_init_gpio_out(&dev->qdev, s->out, 8);

    pl061_reset(s);

    register_savevm(&dev->qdev, "pl061_gpio", -1, 1, pl061_save, pl061_load, s);

    return 0;

}
