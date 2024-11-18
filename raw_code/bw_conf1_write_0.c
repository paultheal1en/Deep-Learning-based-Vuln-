static void bw_conf1_write(void *opaque, target_phys_addr_t addr,

                           uint64_t val, unsigned size)

{

    PCIBus *b = opaque;

    pci_data_write(b, addr, val, size);

}
