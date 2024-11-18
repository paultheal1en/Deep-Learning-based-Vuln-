static AddressSpace *s390_pci_dma_iommu(PCIBus *bus, void *opaque, int devfn)

{

    S390pciState *s = opaque;



    return &s->pbdev[PCI_SLOT(devfn)].as;

}
