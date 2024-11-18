static void watch_mem_write(void *opaque, hwaddr addr,

                            uint64_t val, unsigned size)

{

    check_watchpoint(addr & ~TARGET_PAGE_MASK, size, BP_MEM_WRITE);

    switch (size) {

    case 1:

        stb_phys(&address_space_memory, addr, val);

        break;

    case 2:

        stw_phys(&address_space_memory, addr, val);

        break;

    case 4:

        stl_phys(&address_space_memory, addr, val);

        break;

    default: abort();

    }

}
