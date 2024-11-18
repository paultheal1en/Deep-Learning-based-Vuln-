static int msix_is_masked(PCIDevice *dev, int vector)

{

    unsigned offset =

        vector * PCI_MSIX_ENTRY_SIZE + PCI_MSIX_ENTRY_VECTOR_CTRL;

    return dev->msix_function_masked ||

	   dev->msix_table_page[offset] & PCI_MSIX_ENTRY_CTRL_MASKBIT;

}
