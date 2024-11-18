static void watch_mem_write(void *opaque, target_phys_addr_t addr,

                            uint64_t val, unsigned size)

{

    check_watchpoint(addr & ~TARGET_PAGE_MASK, ~(size - 1), BP_MEM_WRITE);

    switch (size) {

    case 1:

        stb_phys(addr, val);

        break;

    case 2:

        stw_phys(addr, val);

        break;

    case 4:

        stl_phys(addr, val);

        break;

    default: abort();

    }

}
