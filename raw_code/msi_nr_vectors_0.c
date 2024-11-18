static inline unsigned int msi_nr_vectors(uint16_t flags)

{

    return 1U <<

        ((flags & PCI_MSI_FLAGS_QSIZE) >> (ffs(PCI_MSI_FLAGS_QSIZE) - 1));

}
