static void idreg_init(target_phys_addr_t addr)

{

    DeviceState *dev;

    SysBusDevice *s;



    dev = qdev_create(NULL, "macio_idreg");

    qdev_init(dev);

    s = sysbus_from_qdev(dev);



    sysbus_mmio_map(s, 0, addr);

    cpu_physical_memory_write_rom(addr, idreg_data, sizeof(idreg_data));

}
