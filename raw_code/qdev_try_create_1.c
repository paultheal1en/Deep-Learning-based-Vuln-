DeviceState *qdev_try_create(BusState *bus, const char *type)

{

    DeviceState *dev;



    if (object_class_by_name(type) == NULL) {

        return NULL;

    }

    dev = DEVICE(object_new(type));

    if (!dev) {

        return NULL;

    }



    if (!bus) {







        bus = sysbus_get_default();

    }



    qdev_set_parent_bus(dev, bus);

    object_unref(OBJECT(dev));

    return dev;

}