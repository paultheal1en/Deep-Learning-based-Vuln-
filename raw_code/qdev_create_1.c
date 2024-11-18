DeviceState *qdev_create(BusState *bus, const char *name)

{

    DeviceState *dev;



    dev = qdev_try_create(bus, name);

    if (!dev) {

        hw_error("Unknown device '%s' for bus '%s'\n", name, bus->info->name);

    }



    return dev;

}
