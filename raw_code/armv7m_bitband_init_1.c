static void armv7m_bitband_init(void)

{

    DeviceState *dev;



    dev = qdev_create(NULL, "ARM,bitband-memory");

    qdev_prop_set_uint32(dev, "base", 0x20000000);

    qdev_init(dev);

    sysbus_mmio_map(sysbus_from_qdev(dev), 0, 0x22000000);



    dev = qdev_create(NULL, "ARM,bitband-memory");

    qdev_prop_set_uint32(dev, "base", 0x40000000);

    qdev_init(dev);

    sysbus_mmio_map(sysbus_from_qdev(dev), 0, 0x42000000);

}
