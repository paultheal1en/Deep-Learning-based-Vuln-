void qbus_free(BusState *bus)

{

    DeviceState *dev;



    while ((dev = QLIST_FIRST(&bus->children)) != NULL) {

        qdev_free(dev);

    }

    if (bus->parent) {

        QLIST_REMOVE(bus, sibling);

        bus->parent->num_child_bus--;

    }


    if (bus->qdev_allocated) {

        qemu_free(bus);

    }

}