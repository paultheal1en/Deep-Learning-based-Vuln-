static void pc_dimm_plug(HotplugHandler *hotplug_dev,

                         DeviceState *dev, Error **errp)

{

    HotplugHandlerClass *hhc;

    Error *local_err = NULL;

    PCMachineState *pcms = PC_MACHINE(hotplug_dev);

    PCMachineClass *pcmc = PC_MACHINE_GET_CLASS(pcms);

    PCDIMMDevice *dimm = PC_DIMM(dev);

    PCDIMMDeviceClass *ddc = PC_DIMM_GET_CLASS(dimm);

    MemoryRegion *mr = ddc->get_memory_region(dimm);

    uint64_t align = TARGET_PAGE_SIZE;

    bool is_nvdimm = object_dynamic_cast(OBJECT(dev), TYPE_NVDIMM);



    if (memory_region_get_alignment(mr) && pcmc->enforce_aligned_dimm) {

        align = memory_region_get_alignment(mr);

    }



    if (!pcms->acpi_dev) {

        error_setg(&local_err,

                   "memory hotplug is not enabled: missing acpi device");

        goto out;

    }



    if (is_nvdimm && !pcms->acpi_nvdimm_state.is_enabled) {

        error_setg(&local_err,

                   "nvdimm is not enabled: missing 'nvdimm' in '-M'");

        goto out;

    }



    pc_dimm_memory_plug(dev, &pcms->hotplug_memory, mr, align, &local_err);

    if (local_err) {

        goto out;

    }



    if (is_nvdimm) {

        nvdimm_plug(&pcms->acpi_nvdimm_state);

    }



    hhc = HOTPLUG_HANDLER_GET_CLASS(pcms->acpi_dev);

    hhc->plug(HOTPLUG_HANDLER(pcms->acpi_dev), dev, &error_abort);

out:

    error_propagate(errp, local_err);

}
