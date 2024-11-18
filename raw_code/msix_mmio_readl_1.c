static uint32_t msix_mmio_readl(void *opaque, target_phys_addr_t addr)

{

    PCIDevice *dev = opaque;

    unsigned int offset = addr & (MSIX_PAGE_SIZE - 1);

    void *page = dev->msix_table_page;

    uint32_t val = 0;



    memcpy(&val, (void *)((char *)page + offset), 4);



    return val;

}
