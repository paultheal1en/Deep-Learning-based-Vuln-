static uint64_t subpage_ram_read(void *opaque, target_phys_addr_t addr,

                                 unsigned size)

{

    ram_addr_t raddr = addr;

    void *ptr = qemu_get_ram_ptr(raddr);

    switch (size) {

    case 1: return ldub_p(ptr);

    case 2: return lduw_p(ptr);

    case 4: return ldl_p(ptr);

    default: abort();

    }

}
