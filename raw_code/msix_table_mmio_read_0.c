static uint64_t msix_table_mmio_read(void *opaque, target_phys_addr_t addr,

                                     unsigned size)

{

    PCIDevice *dev = opaque;



    return pci_get_long(dev->msix_table + addr);

}
