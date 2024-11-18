fdctrl_t *fdctrl_init_sysbus(qemu_irq irq, int dma_chann,

                             target_phys_addr_t mmio_base,

                             DriveInfo **fds)

{

    fdctrl_t *fdctrl;

    DeviceState *dev;

    fdctrl_sysbus_t *sys;



    dev = qdev_create(NULL, "sysbus-fdc");

    sys = DO_UPCAST(fdctrl_sysbus_t, busdev.qdev, dev);

    fdctrl = &sys->state;

    fdctrl->dma_chann = dma_chann; /* FIXME */

    qdev_prop_set_drive(dev, "driveA", fds[0]);

    qdev_prop_set_drive(dev, "driveB", fds[1]);

    if (qdev_init(dev) != 0)

        return NULL;

    sysbus_connect_irq(&sys->busdev, 0, irq);

    sysbus_mmio_map(&sys->busdev, 0, mmio_base);



    return fdctrl;

}
