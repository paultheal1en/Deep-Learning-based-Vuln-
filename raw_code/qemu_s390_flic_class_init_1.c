static void qemu_s390_flic_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    S390FLICStateClass *fsc = S390_FLIC_COMMON_CLASS(oc);



    dc->reset = qemu_s390_flic_reset;

    fsc->register_io_adapter = qemu_s390_register_io_adapter;

    fsc->io_adapter_map = qemu_s390_io_adapter_map;

    fsc->add_adapter_routes = qemu_s390_add_adapter_routes;

    fsc->release_adapter_routes = qemu_s390_release_adapter_routes;

    fsc->clear_io_irq = qemu_s390_clear_io_flic;

    fsc->modify_ais_mode = qemu_s390_modify_ais_mode;


}