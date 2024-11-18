static void virtio_pci_exit(PCIDevice *pci_dev)

{

    VirtIOPCIProxy *proxy = VIRTIO_PCI(pci_dev);

    virtio_pci_stop_ioeventfd(proxy);

    memory_region_destroy(&proxy->bar);

    msix_uninit_exclusive_bar(pci_dev);

}
