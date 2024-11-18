static void virtio_write_config(PCIDevice *pci_dev, uint32_t address,

                                uint32_t val, int len)

{

    VirtIOPCIProxy *proxy = DO_UPCAST(VirtIOPCIProxy, pci_dev, pci_dev);



    if (PCI_COMMAND == address) {

        if (!(val & PCI_COMMAND_MASTER)) {

            proxy->vdev->status &= !VIRTIO_CONFIG_S_DRIVER_OK;

        }

    }



    pci_default_write_config(pci_dev, address, val, len);

    if(proxy->vdev->nvectors)

        msix_write_config(pci_dev, address, val, len);

}
