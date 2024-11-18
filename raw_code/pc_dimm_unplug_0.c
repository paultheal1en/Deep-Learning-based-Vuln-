static void pc_dimm_unplug(HotplugHandler *hotplug_dev,

                           DeviceState *dev, Error **errp)

{

    PCMachineState *pcms = PC_MACHINE(hotplug_dev);

    PCDIMMDevice *dimm = PC_DIMM(dev);

    PCDIMMDeviceClass *ddc = PC_DIMM_GET_CLASS(dimm);

    MemoryRegion *mr = ddc->get_memory_region(dimm);

    HotplugHandlerClass *hhc;

    Error *local_err = NULL;



    if (object_dynamic_cast(OBJECT(dev), TYPE_NVDIMM)) {

        error_setg(&local_err,

                   "nvdimm device hot unplug is not supported yet.");

        goto out;

    }



    hhc = HOTPLUG_HANDLER_GET_CLASS(pcms->acpi_dev);

    hhc->unplug(HOTPLUG_HANDLER(pcms->acpi_dev), dev, &local_err);



    if (local_err) {

        goto out;

    }



    pc_dimm_memory_unplug(dev, &pcms->hotplug_memory, mr);

    object_unparent(OBJECT(dev));



 out:

    error_propagate(errp, local_err);

}
