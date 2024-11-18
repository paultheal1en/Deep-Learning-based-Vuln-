static DeviceState *apic_init(void *env, uint8_t apic_id)

{

    DeviceState *dev;

    SysBusDevice *d;

    static int apic_mapped;



    dev = qdev_create(NULL, "apic");

    qdev_prop_set_uint8(dev, "id", apic_id);

    qdev_prop_set_ptr(dev, "cpu_env", env);

    qdev_init_nofail(dev);

    d = sysbus_from_qdev(dev);



    /* XXX: mapping more APICs at the same memory location */

    if (apic_mapped == 0) {

        /* NOTE: the APIC is directly connected to the CPU - it is not

           on the global memory bus. */

        /* XXX: what if the base changes? */

        sysbus_mmio_map(d, 0, MSI_ADDR_BASE);

        apic_mapped = 1;

    }



    msi_supported = true;



    return dev;

}
