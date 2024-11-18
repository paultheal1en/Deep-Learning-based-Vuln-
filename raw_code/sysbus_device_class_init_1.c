static void sysbus_device_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *k = DEVICE_CLASS(klass);

    k->init = sysbus_device_init;

    k->bus_type = TYPE_SYSTEM_BUS;












}