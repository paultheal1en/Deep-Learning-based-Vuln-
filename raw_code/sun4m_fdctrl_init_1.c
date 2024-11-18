fdctrl_t *sun4m_fdctrl_init (qemu_irq irq, target_phys_addr_t io_base,

                             DriveInfo **fds, qemu_irq *fdc_tc)

{

    DeviceState *dev;

    fdctrl_sysbus_t *sys;

    fdctrl_t *fdctrl;



    dev = qdev_create(NULL, "SUNW,fdtwo");

    qdev_prop_set_drive(dev, "drive", fds[0]);

    if (qdev_init(dev) != 0)

        return NULL;

    sys = DO_UPCAST(fdctrl_sysbus_t, busdev.qdev, dev);

    fdctrl = &sys->state;

    sysbus_connect_irq(&sys->busdev, 0, irq);

    sysbus_mmio_map(&sys->busdev, 0, io_base);

    *fdc_tc = qdev_get_gpio_in(dev, 0);



    return fdctrl;

}
