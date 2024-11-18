void hmp_pci_add(Monitor *mon, const QDict *qdict)

{

    PCIDevice *dev = NULL;

    const char *pci_addr = qdict_get_str(qdict, "pci_addr");

    const char *type = qdict_get_str(qdict, "type");

    const char *opts = qdict_get_try_str(qdict, "opts");



    /* strip legacy tag */

    if (!strncmp(pci_addr, "pci_addr=", 9)) {

        pci_addr += 9;

    }



    if (!opts) {

        opts = "";

    }



    if (!strcmp(pci_addr, "auto"))

        pci_addr = NULL;



    if (strcmp(type, "nic") == 0) {

        dev = qemu_pci_hot_add_nic(mon, pci_addr, opts);

    } else if (strcmp(type, "storage") == 0) {

        dev = qemu_pci_hot_add_storage(mon, pci_addr, opts);

    } else {

        monitor_printf(mon, "invalid type: %s\n", type);

    }



    if (dev) {

        monitor_printf(mon, "OK root bus %s, bus %d, slot %d, function %d\n",

                       pci_root_bus_path(dev),

                       pci_bus_num(dev->bus), PCI_SLOT(dev->devfn),

                       PCI_FUNC(dev->devfn));

    } else

        monitor_printf(mon, "failed to add %s\n", opts);

}
