static void s390_virtio_bridge_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SysBusDeviceClass *k = SYS_BUS_DEVICE_CLASS(klass);



    k->init = s390_virtio_bridge_init;

    dc->no_user = 1;

}
