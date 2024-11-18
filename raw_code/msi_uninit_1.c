void msi_uninit(struct PCIDevice *dev)

{

    uint16_t flags = pci_get_word(dev->config + msi_flags_off(dev));

    uint8_t cap_size = msi_cap_sizeof(flags);

    pci_del_capability(dev, PCI_CAP_ID_MSIX, cap_size);

    MSI_DEV_PRINTF(dev, "uninit\n");

}
