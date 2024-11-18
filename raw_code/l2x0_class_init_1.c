static void l2x0_class_init(ObjectClass *klass, void *data)

{

    SysBusDeviceClass *k = SYS_BUS_DEVICE_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);



    k->init = l2x0_priv_init;

    dc->vmsd = &vmstate_l2x0;

    dc->no_user = 1;

    dc->props = l2x0_properties;

    dc->reset = l2x0_priv_reset;

}
