static int pci_unin_map_irq(PCIDevice *pci_dev, int irq_num)

{

    int retval;

    int devfn = pci_dev->devfn & 0x00FFFFFF;



    retval = (((devfn >> 11) & 0x1F) + irq_num) & 3;



    return retval;

}
