DeviceState *qdev_create(BusState *bus, const char *name)

{

    DeviceType *t;

    DeviceState *dev;



    for (t = device_type_list; t; t = t->next) {

        if (strcmp(t->info->name, name) == 0) {

            break;

        }

    }

    if (!t) {

        hw_error("Unknown device '%s'\n", name);

    }



    dev = qemu_mallocz(t->info->size);

    dev->type = t;



    if (!bus) {

        /* ???: This assumes system busses have no additional state.  */

        if (!main_system_bus) {

            main_system_bus = qbus_create(BUS_TYPE_SYSTEM, sizeof(BusState),

                                          NULL, "main-system-bus");

        }

        bus = main_system_bus;

    }

    if (t->info->bus_type != bus->type) {

        /* TODO: Print bus type names.  */

        hw_error("Device '%s' on wrong bus type (%d/%d)", name,

                 t->info->bus_type, bus->type);

    }

    dev->parent_bus = bus;

    LIST_INSERT_HEAD(&bus->children, dev, sibling);

    return dev;

}
