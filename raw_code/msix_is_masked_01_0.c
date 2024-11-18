static int msix_is_masked(PCIDevice *dev, int vector)

{

    unsigned offset = vector * MSIX_ENTRY_SIZE + MSIX_VECTOR_CTRL;

    return dev->msix_table_page[offset] & MSIX_VECTOR_MASK;

}
