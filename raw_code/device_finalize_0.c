static void device_finalize(Object *obj)

{

    DeviceState *dev = DEVICE(obj);

    BusState *bus;

    DeviceClass *dc = DEVICE_GET_CLASS(dev);



    if (dev->realized) {

        while (dev->num_child_bus) {

            bus = QLIST_FIRST(&dev->child_bus);

            qbus_free(bus);

        }

        if (qdev_get_vmsd(dev)) {

            vmstate_unregister(dev, qdev_get_vmsd(dev), dev);

        }

        if (dc->exit) {

            dc->exit(dev);

        }

        if (dev->opts) {

            qemu_opts_del(dev->opts);

        }

    }

}
