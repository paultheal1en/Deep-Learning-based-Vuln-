static DeviceState *qbus_find_dev(BusState *bus, char *elem)

{

    DeviceState *dev;



    /*

     * try to match in order:

     *   (1) instance id, if present

     *   (2) driver name

     *   (3) driver alias, if present

     */

    LIST_FOREACH(dev, &bus->children, sibling) {

        if (dev->id  &&  strcmp(dev->id, elem) == 0) {

            return dev;

        }

    }

    LIST_FOREACH(dev, &bus->children, sibling) {

        if (strcmp(dev->info->name, elem) == 0) {

            return dev;

        }

    }

    LIST_FOREACH(dev, &bus->children, sibling) {

        if (dev->info->alias && strcmp(dev->info->alias, elem) == 0) {

            return dev;

        }

    }

    return NULL;

}
