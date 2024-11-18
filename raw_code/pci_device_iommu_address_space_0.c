AddressSpace *pci_device_iommu_address_space(PCIDevice *dev)

{

    PCIBus *bus = PCI_BUS(dev->bus);

    PCIBus *iommu_bus = bus;



    while(iommu_bus && !iommu_bus->iommu_fn && iommu_bus->parent_dev) {

        iommu_bus = PCI_BUS(iommu_bus->parent_dev->bus);

    }

    if (iommu_bus && iommu_bus->iommu_fn) {

        return iommu_bus->iommu_fn(bus, iommu_bus->iommu_opaque, dev->devfn);

    }

    return &address_space_memory;

}
