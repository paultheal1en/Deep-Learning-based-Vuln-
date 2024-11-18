void acpi_memory_plug_cb(ACPIREGS *ar, qemu_irq irq, MemHotplugState *mem_st,

                         DeviceState *dev, Error **errp)

{

    MemStatus *mdev;



    mdev = acpi_memory_slot_status(mem_st, dev, errp);

    if (!mdev) {

        return;

    }



    mdev->dimm = dev;

    mdev->is_enabled = true;

    mdev->is_inserting = true;



    /* do ACPI magic */

    acpi_send_gpe_event(ar, irq, ACPI_MEMORY_HOTPLUG_STATUS);

    return;

}
