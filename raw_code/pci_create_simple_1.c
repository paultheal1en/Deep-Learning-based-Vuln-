PCIDevice *pci_create_simple(PCIBus *bus, int devfn, const char *name)

{

    PCIDevice *dev = pci_create(bus, devfn, name);

    qdev_init(&dev->qdev);

    return dev;

}
