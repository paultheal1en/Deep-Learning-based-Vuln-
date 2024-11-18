static void special_write(void *opaque, target_phys_addr_t addr,

                          uint64_t val, unsigned size)

{

    qemu_log("pci: special write cycle");

}
