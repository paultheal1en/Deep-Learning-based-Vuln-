static void nvdimm_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    PCDIMMDeviceClass *ddc = PC_DIMM_CLASS(oc);

    NVDIMMClass *nvc = NVDIMM_CLASS(oc);



    /* nvdimm hotplug has not been supported yet. */

    dc->hotpluggable = false;



    ddc->realize = nvdimm_realize;

    ddc->get_memory_region = nvdimm_get_memory_region;

    ddc->get_vmstate_memory_region = nvdimm_get_vmstate_memory_region;



    nvc->read_label_data = nvdimm_read_label_data;

    nvc->write_label_data = nvdimm_write_label_data;

}
