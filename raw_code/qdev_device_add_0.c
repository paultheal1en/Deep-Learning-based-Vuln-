DeviceState *qdev_device_add(QemuOpts *opts)

{

    DeviceClass *k;

    const char *driver, *path, *id;

    DeviceState *qdev;

    BusState *bus;



    driver = qemu_opt_get(opts, "driver");

    if (!driver) {

        qerror_report(QERR_MISSING_PARAMETER, "driver");

        return NULL;

    }



    /* find driver */

    k = DEVICE_CLASS(object_class_by_name(driver));



    /* find bus */

    path = qemu_opt_get(opts, "bus");

    if (path != NULL) {

        bus = qbus_find(path);

        if (!bus) {

            return NULL;

        }

        if (bus->info != k->bus_info) {

            qerror_report(QERR_BAD_BUS_FOR_DEVICE,

                           driver, bus->info->name);

            return NULL;

        }

    } else {

        bus = qbus_find_recursive(main_system_bus, NULL, k->bus_info);

        if (!bus) {

            qerror_report(QERR_NO_BUS_FOR_DEVICE,

                          driver, k->bus_info->name);

            return NULL;

        }

    }

    if (qdev_hotplug && !bus->allow_hotplug) {

        qerror_report(QERR_BUS_NO_HOTPLUG, bus->name);

        return NULL;

    }



    /* create device, set properties */

    qdev = qdev_create_from_info(bus, driver);

    id = qemu_opts_id(opts);

    if (id) {

        qdev->id = id;

        qdev_property_add_child(qdev_get_peripheral(), qdev->id, qdev, NULL);

    } else {

        static int anon_count;

        gchar *name = g_strdup_printf("device[%d]", anon_count++);

        qdev_property_add_child(qdev_get_peripheral_anon(), name,

                                qdev, NULL);

        g_free(name);

    }        

    if (qemu_opt_foreach(opts, set_property, qdev, 1) != 0) {

        qdev_free(qdev);

        return NULL;

    }

    if (qdev_init(qdev) < 0) {

        qerror_report(QERR_DEVICE_INIT_FAILED, driver);

        return NULL;

    }

    qdev->opts = opts;

    return qdev;

}
