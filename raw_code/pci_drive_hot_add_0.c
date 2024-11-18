int pci_drive_hot_add(Monitor *mon, const QDict *qdict, DriveInfo *dinfo)

{

    int dom, pci_bus;

    unsigned slot;

    PCIDevice *dev;

    const char *pci_addr = qdict_get_str(qdict, "pci_addr");



    switch (dinfo->type) {

    case IF_SCSI:

        if (pci_read_devaddr(mon, pci_addr, &dom, &pci_bus, &slot)) {

            goto err;

        }

        dev = pci_find_device(pci_find_root_bus(dom), pci_bus,

                              PCI_DEVFN(slot, 0));

        if (!dev) {

            monitor_printf(mon, "no pci device with address %s\n", pci_addr);

            goto err;

        }

        if (scsi_hot_add(mon, &dev->qdev, dinfo, 1) != 0) {

            goto err;

        }

        break;

    default:

        monitor_printf(mon, "Can't hot-add drive to type %d\n", dinfo->type);

        goto err;

    }



    return 0;

err:

    return -1;

}
