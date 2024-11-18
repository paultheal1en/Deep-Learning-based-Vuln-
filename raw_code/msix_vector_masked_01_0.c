static bool msix_vector_masked(PCIDevice *dev, unsigned int vector, bool fmask)

{

    unsigned offset = vector * PCI_MSIX_ENTRY_SIZE;

    uint32_t *data = (uint32_t *)&dev->msix_table[offset + PCI_MSIX_ENTRY_DATA];

    /* MSIs on Xen can be remapped into pirqs. In those cases, masking

     * and unmasking go through the PV evtchn path. */

    if (xen_is_pirq_msi(*data)) {

        return false;

    }

    return fmask || dev->msix_table[offset + PCI_MSIX_ENTRY_VECTOR_CTRL] &

        PCI_MSIX_ENTRY_CTRL_MASKBIT;

}
