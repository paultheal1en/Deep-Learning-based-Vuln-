PCIDevice *pci_try_create(PCIBus *bus, int devfn, const char *name)

{

    return pci_try_create_multifunction(bus, devfn, false, name);

}
