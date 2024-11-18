static BusState *qbus_find_recursive(BusState *bus, const char *name,

                                     const char *bus_typename)

{

    BusChild *kid;

    BusState *child, *ret;

    int match = 1;



    if (name && (strcmp(bus->name, name) != 0)) {

        match = 0;

    }

    if (bus_typename &&

        (strcmp(object_get_typename(OBJECT(bus)), bus_typename) != 0)) {

        match = 0;

    }

    if (match) {

        return bus;

    }



    QTAILQ_FOREACH(kid, &bus->children, sibling) {

        DeviceState *dev = kid->child;

        QLIST_FOREACH(child, &dev->child_bus, sibling) {

            ret = qbus_find_recursive(child, name, bus_typename);

            if (ret) {

                return ret;

            }

        }

    }

    return NULL;

}
