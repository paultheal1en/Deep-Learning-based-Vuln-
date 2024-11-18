MemoryRegion *pci_address_space_io(PCIDevice *dev)

{

    return dev->bus->address_space_io;

}
