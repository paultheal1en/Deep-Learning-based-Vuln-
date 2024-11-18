DeviceState *exynos4210_uart_create(target_phys_addr_t addr,

                                 int fifo_size,

                                 int channel,

                                 CharDriverState *chr,

                                 qemu_irq irq)

{

    DeviceState  *dev;

    SysBusDevice *bus;



    const char chr_name[] = "serial";

    char label[ARRAY_SIZE(chr_name) + 1];



    dev = qdev_create(NULL, "exynos4210.uart");



    if (!chr) {

        if (channel >= MAX_SERIAL_PORTS) {

            hw_error("Only %d serial ports are supported by QEMU.\n",

                     MAX_SERIAL_PORTS);

        }

        chr = serial_hds[channel];

        if (!chr) {

            snprintf(label, ARRAY_SIZE(label), "%s%d", chr_name, channel);

            chr = qemu_chr_new(label, "null", NULL);

            if (!(chr)) {

                hw_error("Can't assign serial port to UART%d.\n", channel);

            }

        }

    }



    qdev_prop_set_chr(dev, "chardev", chr);

    qdev_prop_set_uint32(dev, "channel", channel);

    qdev_prop_set_uint32(dev, "rx-size", fifo_size);

    qdev_prop_set_uint32(dev, "tx-size", fifo_size);



    bus = sysbus_from_qdev(dev);

    qdev_init_nofail(dev);

    if (addr != (target_phys_addr_t)-1) {

        sysbus_mmio_map(bus, 0, addr);

    }

    sysbus_connect_irq(bus, 0, irq);



    return dev;

}
