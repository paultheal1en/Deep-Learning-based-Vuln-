void empty_slot_init(target_phys_addr_t addr, uint64_t slot_size)

{

    if (slot_size > 0) {

        /* Only empty slots larger than 0 byte need handling. */

        DeviceState *dev;

        SysBusDevice *s;

        EmptySlot *e;



        dev = qdev_create(NULL, "empty_slot");

        s = sysbus_from_qdev(dev);

        e = FROM_SYSBUS(EmptySlot, s);

        e->size = slot_size;



        qdev_init_nofail(dev);



        sysbus_mmio_map(s, 0, addr);

    }

}
