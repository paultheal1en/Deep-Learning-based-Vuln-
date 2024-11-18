static PCIDevice *qemu_pci_hot_add_storage(Monitor *mon,

                                           const char *devaddr,

                                           const char *opts)

{

    PCIDevice *dev;

    DriveInfo *dinfo = NULL;

    int type = -1;

    char buf[128];

    PCIBus *bus;

    int devfn;



    if (get_param_value(buf, sizeof(buf), "if", opts)) {

        if (!strcmp(buf, "scsi"))

            type = IF_SCSI;

        else if (!strcmp(buf, "virtio")) {

            type = IF_VIRTIO;

        } else {

            monitor_printf(mon, "type %s not a hotpluggable PCI device.\n", buf);

            return NULL;

        }

    } else {

        monitor_printf(mon, "no if= specified\n");

        return NULL;

    }



    if (get_param_value(buf, sizeof(buf), "file", opts)) {

        dinfo = add_init_drive(opts);

        if (!dinfo)

            return NULL;

        if (dinfo->devaddr) {

            monitor_printf(mon, "Parameter addr not supported\n");

            return NULL;

        }

    } else {

        dinfo = NULL;

    }



    bus = pci_get_bus_devfn(&devfn, devaddr);

    if (!bus) {

        monitor_printf(mon, "Invalid PCI device address %s\n", devaddr);

        return NULL;

    }



    switch (type) {

    case IF_SCSI:

        if (!dinfo) {

            monitor_printf(mon, "scsi requires a backing file/device.\n");

            return NULL;

        }

        dev = pci_create(bus, devfn, "lsi53c895a");

        if (qdev_init(&dev->qdev) < 0)

            dev = NULL;

        if (dev) {

            BusState *scsibus = QLIST_FIRST(&dev->qdev.child_bus);

            scsi_bus_legacy_add_drive(DO_UPCAST(SCSIBus, qbus, scsibus),

                                      dinfo, dinfo->unit);

        }

        break;

    case IF_VIRTIO:

        if (!dinfo) {

            monitor_printf(mon, "virtio requires a backing file/device.\n");

            return NULL;

        }

        dev = pci_create(bus, devfn, "virtio-blk-pci");

        qdev_prop_set_drive(&dev->qdev, "drive", dinfo);

        if (qdev_init(&dev->qdev) < 0)

            dev = NULL;

        break;

    default:

        dev = NULL;

    }

    return dev;

}
