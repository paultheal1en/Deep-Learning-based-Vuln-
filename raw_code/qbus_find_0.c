static BusState *qbus_find(const char *path)

{

    DeviceState *dev;

    BusState *bus;

    char elem[128], msg[256];

    int pos, len;



    /* find start element */

    if (path[0] == '/') {

        bus = main_system_bus;

        pos = 0;

    } else {

        if (sscanf(path, "%127[^/]%n", elem, &len) != 1) {

            qemu_error("path parse error (\"%s\")\n", path);

            return NULL;

        }

        bus = qbus_find_recursive(main_system_bus, elem, NULL);

        if (!bus) {

            qemu_error("bus \"%s\" not found\n", elem);

            return NULL;

        }

        pos = len;

    }



    for (;;) {

        if (path[pos] == '\0') {

            /* we are done */

            return bus;

        }



        /* find device */

        if (sscanf(path+pos, "/%127[^/]%n", elem, &len) != 1) {

            qemu_error("path parse error (\"%s\" pos %d)\n", path, pos);

            return NULL;

        }

        pos += len;

        dev = qbus_find_dev(bus, elem);

        if (!dev) {

            qbus_list_dev(bus, msg, sizeof(msg));

            qemu_error("device \"%s\" not found\n%s\n", elem, msg);

            return NULL;

        }

        if (path[pos] == '\0') {

            /* last specified element is a device.  If it has exactly

             * one child bus accept it nevertheless */

            switch (dev->num_child_bus) {

            case 0:

                qemu_error("device has no child bus (%s)\n", path);

                return NULL;

            case 1:

                return LIST_FIRST(&dev->child_bus);

            default:

                qbus_list_bus(dev, msg, sizeof(msg));

                qemu_error("device has multiple child busses (%s)\n%s\n",

                           path, msg);

                return NULL;

            }

        }



        /* find bus */

        if (sscanf(path+pos, "/%127[^/]%n", elem, &len) != 1) {

            qemu_error("path parse error (\"%s\" pos %d)\n", path, pos);

            return NULL;

        }

        pos += len;

        bus = qbus_find_bus(dev, elem);

        if (!bus) {

            qbus_list_bus(dev, msg, sizeof(msg));

            qemu_error("child bus \"%s\" not found\n%s\n", elem, msg);

            return NULL;

        }

    }

}
