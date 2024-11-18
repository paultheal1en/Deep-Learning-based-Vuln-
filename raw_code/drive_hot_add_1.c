void drive_hot_add(Monitor *mon, const QDict *qdict)

{

    int dom, pci_bus;

    unsigned slot;

    int type, bus;

    PCIDevice *dev;

    DriveInfo *dinfo = NULL;

    const char *pci_addr = qdict_get_str(qdict, "pci_addr");

    const char *opts = qdict_get_str(qdict, "opts");

    BusState *scsibus;



    dinfo = add_init_drive(opts);

    if (!dinfo)

        goto err;

    if (dinfo->devaddr) {

        monitor_printf(mon, "Parameter addr not supported\n");

        goto err;

    }

    type = dinfo->type;

    bus = drive_get_max_bus (type);



    switch (type) {

    case IF_SCSI:

        if (pci_read_devaddr(mon, pci_addr, &dom, &pci_bus, &slot)) {

            goto err;

        }

        dev = pci_find_device(pci_bus, slot, 0);

        if (!dev) {

            monitor_printf(mon, "no pci device with address %s\n", pci_addr);

            goto err;

        }

        scsibus = QLIST_FIRST(&dev->qdev.child_bus);

        scsi_bus_legacy_add_drive(DO_UPCAST(SCSIBus, qbus, scsibus),

                                  dinfo, dinfo->unit);

        monitor_printf(mon, "OK bus %d, unit %d\n",

                       dinfo->bus,

                       dinfo->unit);

        break;

    case IF_NONE:

        monitor_printf(mon, "OK\n");

        break;

    default:

        monitor_printf(mon, "Can't hot-add drive to type %d\n", type);

        goto err;

    }

    return;



err:

    if (dinfo)

        drive_uninit(dinfo);

    return;

}
