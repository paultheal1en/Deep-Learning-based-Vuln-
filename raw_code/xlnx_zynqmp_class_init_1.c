static void xlnx_zynqmp_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    dc->props = xlnx_zynqmp_props;
    dc->realize = xlnx_zynqmp_realize;
}