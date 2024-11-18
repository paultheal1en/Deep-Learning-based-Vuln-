static bool msi_is_masked(const PCIDevice *dev, unsigned int vector)

{

    uint16_t flags = pci_get_word(dev->config + msi_flags_off(dev));

    uint32_t mask;

    assert(vector < PCI_MSI_VECTORS_MAX);



    if (!(flags & PCI_MSI_FLAGS_MASKBIT)) {

        return false;

    }



    mask = pci_get_long(dev->config +

                        msi_mask_off(dev, flags & PCI_MSI_FLAGS_64BIT));

    return mask & (1U << vector);

}
