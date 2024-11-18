static void qdev_print(Monitor *mon, DeviceState *dev, int indent)

{

    BusState *child;

    qdev_printf("dev: %s, id \"%s\"\n", dev->info->name,

                dev->id ? dev->id : "");

    indent += 2;

    if (dev->num_gpio_in) {

        qdev_printf("gpio-in %d\n", dev->num_gpio_in);

    }

    if (dev->num_gpio_out) {

        qdev_printf("gpio-out %d\n", dev->num_gpio_out);

    }

    qdev_print_props(mon, dev, dev->info->props, "dev", indent);

    qdev_print_props(mon, dev, dev->parent_bus->info->props, "bus", indent);

    if (dev->parent_bus->info->print_dev)

        dev->parent_bus->info->print_dev(mon, dev, indent);

    LIST_FOREACH(child, &dev->child_bus, sibling) {

        qbus_print(mon, child, indent);

    }

}
