static AddressSpace *virtio_pci_get_dma_as(DeviceState *d)

{

    VirtIOPCIProxy *proxy = VIRTIO_PCI(d);

    PCIDevice *dev = &proxy->pci_dev;



    return pci_get_address_space(dev);

}
