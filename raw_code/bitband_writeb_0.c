static void bitband_writeb(void *opaque, target_phys_addr_t offset,

                           uint32_t value)

{

    uint32_t addr;

    uint8_t mask;

    uint8_t v;

    addr = bitband_addr(opaque, offset);

    mask = (1 << ((offset >> 2) & 7));

    cpu_physical_memory_read(addr, &v, 1);

    if (value & 1)

        v |= mask;

    else

        v &= ~mask;

    cpu_physical_memory_write(addr, &v, 1);

}
