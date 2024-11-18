static int pci_apb_map_irq(PCIDevice *pci_dev, int irq_num)

{

    return ((pci_dev->devfn & 0x18) >> 1) + irq_num;

}
