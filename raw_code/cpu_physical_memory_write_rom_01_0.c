void cpu_physical_memory_write_rom(hwaddr addr,

                                   const uint8_t *buf, int len)

{

    hwaddr l;

    uint8_t *ptr;

    hwaddr addr1;

    MemoryRegion *mr;



    while (len > 0) {

        l = len;

        mr = address_space_translate(&address_space_memory,

                                     addr, &addr1, &l, true);



        if (!(memory_region_is_ram(mr) ||

              memory_region_is_romd(mr))) {

            /* do nothing */

        } else {

            addr1 += memory_region_get_ram_addr(mr);

            /* ROM/RAM case */

            ptr = qemu_get_ram_ptr(addr1);

            memcpy(ptr, buf, l);

            invalidate_and_set_dirty(addr1, l);

        }

        len -= l;

        buf += l;

        addr += l;

    }

}
