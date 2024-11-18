static void pc_dimm_plug(HotplugHandler *hotplug_dev,

                         DeviceState *dev, Error **errp)

{

    HotplugHandlerClass *hhc;

    Error *local_err = NULL;

    PCMachineState *pcms = PC_MACHINE(hotplug_dev);

    PCDIMMDevice *dimm = PC_DIMM(dev);

    PCDIMMDeviceClass *ddc = PC_DIMM_GET_CLASS(dimm);

    MemoryRegion *mr = ddc->get_memory_region(dimm);

    uint64_t align = TARGET_PAGE_SIZE;



    if (memory_region_get_alignment(mr) && pcms->enforce_aligned_dimm) {

        align = memory_region_get_alignment(mr);

    }



    if (!pcms->acpi_dev) {

        error_setg(&local_err,

                   "memory hotplug is not enabled: missing acpi device");

        goto out;

    }



    pc_dimm_memory_plug(dev, &pcms->hotplug_memory, mr, align, &local_err);

    if (local_err) {

        goto out;

    }



    hhc = HOTPLUG_HANDLER_GET_CLASS(pcms->acpi_dev);

    hhc->plug(HOTPLUG_HANDLER(pcms->acpi_dev), dev, &error_abort);

out:

    error_propagate(errp, local_err);

}
