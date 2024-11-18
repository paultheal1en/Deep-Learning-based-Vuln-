static void virtio_pci_device_unplugged(DeviceState *d)

{

    PCIDevice *pci_dev = PCI_DEVICE(d);

    VirtIOPCIProxy *proxy = VIRTIO_PCI(d);



    virtio_pci_stop_ioeventfd(proxy);

    msix_uninit_exclusive_bar(pci_dev);

}
