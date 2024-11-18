static void subpage_ram_write(void *opaque, target_phys_addr_t addr,

                              uint64_t value, unsigned size)

{

    ram_addr_t raddr = addr;

    void *ptr = qemu_get_ram_ptr(raddr);

    switch (size) {

    case 1: return stb_p(ptr, value);

    case 2: return stw_p(ptr, value);

    case 4: return stl_p(ptr, value);

    default: abort();

    }

}
