static void msix_mmio_writel(void *opaque, target_phys_addr_t addr,

                             uint32_t val)

{

    PCIDevice *dev = opaque;

    unsigned int offset = addr & (MSIX_PAGE_SIZE - 1);

    int vector = offset / MSIX_ENTRY_SIZE;

    memcpy(dev->msix_table_page + offset, &val, 4);

    if (!msix_is_masked(dev, vector) && msix_is_pending(dev, vector)) {

        msix_clr_pending(dev, vector);

        msix_notify(dev, vector);

    }

}
