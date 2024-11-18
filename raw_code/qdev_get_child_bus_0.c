BusState *qdev_get_child_bus(DeviceState *dev, const char *name)

{

    BusState *bus;



    LIST_FOREACH(bus, &dev->child_bus, sibling) {

        if (strcmp(name, bus->name) == 0) {

            return bus;

        }

    }

    return NULL;

}
