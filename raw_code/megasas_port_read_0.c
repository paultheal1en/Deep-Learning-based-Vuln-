static uint64_t megasas_port_read(void *opaque, target_phys_addr_t addr,

                                  unsigned size)

{

    return megasas_mmio_read(opaque, addr & 0xff, size);

}
