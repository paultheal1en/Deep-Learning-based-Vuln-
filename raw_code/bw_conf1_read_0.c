static uint64_t bw_conf1_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    PCIBus *b = opaque;

    return pci_data_read(b, addr, size);

}
