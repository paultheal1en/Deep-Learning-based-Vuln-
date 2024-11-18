AddressSpace *pci_device_iommu_address_space(PCIDevice *dev)

{

    PCIBus *bus = PCI_BUS(dev->bus);



    if (bus->iommu_fn) {

        return bus->iommu_fn(bus, bus->iommu_opaque, dev->devfn);

    }



    if (bus->parent_dev) {

        /** We are ignoring the bus master DMA bit of the bridge

         *  as it would complicate things such as VFIO for no good reason */

        return pci_device_iommu_address_space(bus->parent_dev);

    }



    return &address_space_memory;

}
