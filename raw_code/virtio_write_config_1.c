static void virtio_write_config(PCIDevice *pci_dev, uint32_t address,

                                uint32_t val, int len)

{

    pci_default_write_config(pci_dev, address, val, len);

    msix_write_config(pci_dev, address, val, len);

}
