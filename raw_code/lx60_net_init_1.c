static void lx60_net_init(MemoryRegion *address_space,

        hwaddr base,

        hwaddr descriptors,

        hwaddr buffers,

        qemu_irq irq, NICInfo *nd)

{

    DeviceState *dev;

    SysBusDevice *s;

    MemoryRegion *ram;



    dev = qdev_create(NULL, "open_eth");

    qdev_set_nic_properties(dev, nd);

    qdev_init_nofail(dev);



    s = SYS_BUS_DEVICE(dev);

    sysbus_connect_irq(s, 0, irq);

    memory_region_add_subregion(address_space, base,

            sysbus_mmio_get_region(s, 0));

    memory_region_add_subregion(address_space, descriptors,

            sysbus_mmio_get_region(s, 1));



    ram = g_malloc(sizeof(*ram));

    memory_region_init_ram(ram, OBJECT(s), "open_eth.ram", 16384, &error_abort);

    vmstate_register_ram_global(ram);

    memory_region_add_subregion(address_space, buffers, ram);

}
