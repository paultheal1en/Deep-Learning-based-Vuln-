void acpi_memory_unplug_cb(MemHotplugState *mem_st,

                           DeviceState *dev, Error **errp)

{

    MemStatus *mdev;



    mdev = acpi_memory_slot_status(mem_st, dev, errp);

    if (!mdev) {

        return;

    }



    /* nvdimm device hot unplug is not supported yet. */

    assert(!object_dynamic_cast(OBJECT(dev), TYPE_NVDIMM));

    mdev->is_enabled = false;

    mdev->dimm = NULL;

}
