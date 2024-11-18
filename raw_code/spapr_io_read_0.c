static uint64_t spapr_io_read(void *opaque, hwaddr addr,

                              unsigned size)

{

    switch (size) {

    case 1:

        return cpu_inb(addr);

    case 2:

        return cpu_inw(addr);

    case 4:

        return cpu_inl(addr);

    }

    assert(0);

}
