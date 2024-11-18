static uint64_t bw_io_read(void *opaque, target_phys_addr_t addr, unsigned size)

{

    switch (size) {

    case 1:

        return cpu_inb(addr);

    case 2:

        return cpu_inw(addr);

    case 4:

        return cpu_inl(addr);

    }

    abort();

}
