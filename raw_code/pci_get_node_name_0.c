static void pci_get_node_name(char *nodename, int len, PCIDevice *dev)

{

    int slot = PCI_SLOT(dev->devfn);

    int func = PCI_FUNC(dev->devfn);

    uint32_t ccode = pci_default_read_config(dev, PCI_CLASS_PROG, 3);

    const char *name;



    name = pci_find_device_name((ccode >> 16) & 0xff, (ccode >> 8) & 0xff,

                                ccode & 0xff);



    if (func != 0) {

        snprintf(nodename, len, "%s@%x,%x", name, slot, func);

    } else {

        snprintf(nodename, len, "%s@%x", name, slot);

    }

}
