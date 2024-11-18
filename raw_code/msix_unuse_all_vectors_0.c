void msix_unuse_all_vectors(PCIDevice *dev)

{

    if (!(dev->cap_present & QEMU_PCI_CAP_MSIX))

        return;

    msix_free_irq_entries(dev);

}
