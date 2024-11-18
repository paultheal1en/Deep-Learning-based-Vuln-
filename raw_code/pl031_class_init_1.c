static void pl031_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SysBusDeviceClass *k = SYS_BUS_DEVICE_CLASS(klass);



    k->init = pl031_init;

    dc->no_user = 1;

    dc->vmsd = &vmstate_pl031;

}
