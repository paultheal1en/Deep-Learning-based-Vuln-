DeviceState *qdev_try_create(BusState *bus, const char *name)

{

    DeviceState *dev;






    dev = DEVICE(object_new(name));

    if (!dev) {





    if (!bus) {

        bus = sysbus_get_default();




    qdev_set_parent_bus(dev, bus);

    qdev_prop_set_globals(dev);



    return dev;
