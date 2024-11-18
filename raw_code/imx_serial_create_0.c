void imx_serial_create(int uart, const target_phys_addr_t addr, qemu_irq irq)

{

    DeviceState *dev;

    SysBusDevice *bus;

    CharDriverState *chr;

    const char chr_name[] = "serial";

    char label[ARRAY_SIZE(chr_name) + 1];



    dev = qdev_create(NULL, "imx-serial");



    if (uart >= MAX_SERIAL_PORTS) {

        hw_error("Cannot assign uart %d: QEMU supports only %d ports\n",

                 uart, MAX_SERIAL_PORTS);

    }

    chr = serial_hds[uart];

    if (!chr) {

        snprintf(label, ARRAY_SIZE(label), "%s%d", chr_name, uart);

        chr = qemu_chr_new(label, "null", NULL);

        if (!(chr)) {

            hw_error("Can't assign serial port to imx-uart%d.\n", uart);

        }

    }



    qdev_prop_set_chr(dev, "chardev", chr);

    bus = sysbus_from_qdev(dev);

    qdev_init_nofail(dev);

    if (addr != (target_phys_addr_t)-1) {

        sysbus_mmio_map(bus, 0, addr);

    }

    sysbus_connect_irq(bus, 0, irq);



}
