static int pci_slot_get_pirq(PCIDevice *pci_dev, int irq_num)

{

    int slot_addend;

    slot_addend = (pci_dev->devfn >> 3) - 1;

    return (irq_num + slot_addend) & 3;

}
