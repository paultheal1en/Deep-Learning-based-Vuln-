static void virtio_pci_device_unplugged(DeviceState *d)

{

    VirtIOPCIProxy *proxy = VIRTIO_PCI(d);

    bool modern = !(proxy->flags & VIRTIO_PCI_FLAG_DISABLE_MODERN);

    bool modern_pio = proxy->flags & VIRTIO_PCI_FLAG_MODERN_PIO_NOTIFY;



    virtio_pci_stop_ioeventfd(proxy);



    if (modern) {

        virtio_pci_modern_mem_region_unmap(proxy, &proxy->common);

        virtio_pci_modern_mem_region_unmap(proxy, &proxy->isr);

        virtio_pci_modern_mem_region_unmap(proxy, &proxy->device);

        virtio_pci_modern_mem_region_unmap(proxy, &proxy->notify);

        if (modern_pio) {

            virtio_pci_modern_io_region_unmap(proxy, &proxy->notify_pio);

        }

    }

}
