xen_igd_passthrough_isa_bridge_create(XenPCIPassthroughState *s,

                                      XenHostPCIDevice *dev)

{

    uint16_t gpu_dev_id;

    PCIDevice *d = &s->dev;



    gpu_dev_id = dev->device_id;

    igd_passthrough_isa_bridge_create(d->bus, gpu_dev_id);

}
