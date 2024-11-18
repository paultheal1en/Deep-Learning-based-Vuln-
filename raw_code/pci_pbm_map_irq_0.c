static int pci_pbm_map_irq(PCIDevice *pci_dev, int irq_num)

{

    int bus_offset;

    if (pci_dev->devfn & 1)

        bus_offset = 16;

    else

        bus_offset = 0;

    return (bus_offset + (PCI_SLOT(pci_dev->devfn) << 2) + irq_num) & 0x1f;

}
