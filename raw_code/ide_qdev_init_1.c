static int ide_qdev_init(DeviceState *qdev, DeviceInfo *base)

{

    IDEDevice *dev = DO_UPCAST(IDEDevice, qdev, qdev);

    IDEDeviceInfo *info = DO_UPCAST(IDEDeviceInfo, qdev, base);

    IDEBus *bus = DO_UPCAST(IDEBus, qbus, qdev->parent_bus);



    if (!dev->conf.bs) {

        fprintf(stderr, "%s: no drive specified\n", qdev->info->name);

        goto err;

    }

    if (dev->unit == -1) {

        dev->unit = bus->master ? 1 : 0;

    }

    switch (dev->unit) {

    case 0:

        if (bus->master) {

            fprintf(stderr, "ide: tried to assign master twice\n");

            goto err;

        }

        bus->master = dev;

        break;

    case 1:

        if (bus->slave) {

            fprintf(stderr, "ide: tried to assign slave twice\n");

            goto err;

        }

        bus->slave = dev;

        break;

    default:

        goto err;

    }

    return info->init(dev);



err:

    return -1;

}
