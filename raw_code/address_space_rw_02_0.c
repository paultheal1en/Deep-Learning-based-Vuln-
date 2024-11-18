bool address_space_rw(AddressSpace *as, hwaddr addr, uint8_t *buf,

                      int len, bool is_write)

{

    hwaddr l;

    uint8_t *ptr;

    uint64_t val;

    hwaddr addr1;

    MemoryRegion *mr;

    bool error = false;



    while (len > 0) {

        l = len;

        mr = address_space_translate(as, addr, &addr1, &l, is_write);



        if (is_write) {

            if (!memory_access_is_direct(mr, is_write)) {

                l = memory_access_size(mr, l, addr1);

                /* XXX: could force current_cpu to NULL to avoid

                   potential bugs */

                if (l == 4) {

                    /* 32 bit write access */

                    val = ldl_p(buf);

                    error |= io_mem_write(mr, addr1, val, 4);

                } else if (l == 2) {

                    /* 16 bit write access */

                    val = lduw_p(buf);

                    error |= io_mem_write(mr, addr1, val, 2);

                } else {

                    /* 8 bit write access */

                    val = ldub_p(buf);

                    error |= io_mem_write(mr, addr1, val, 1);

                }

            } else {

                addr1 += memory_region_get_ram_addr(mr);

                /* RAM case */

                ptr = qemu_get_ram_ptr(addr1);

                memcpy(ptr, buf, l);

                invalidate_and_set_dirty(addr1, l);

            }

        } else {

            if (!memory_access_is_direct(mr, is_write)) {

                /* I/O case */

                l = memory_access_size(mr, l, addr1);

                if (l == 4) {

                    /* 32 bit read access */

                    error |= io_mem_read(mr, addr1, &val, 4);

                    stl_p(buf, val);

                } else if (l == 2) {

                    /* 16 bit read access */

                    error |= io_mem_read(mr, addr1, &val, 2);

                    stw_p(buf, val);

                } else {

                    /* 8 bit read access */

                    error |= io_mem_read(mr, addr1, &val, 1);

                    stb_p(buf, val);

                }

            } else {

                /* RAM case */

                ptr = qemu_get_ram_ptr(mr->ram_addr + addr1);

                memcpy(buf, ptr, l);

            }

        }

        len -= l;

        buf += l;

        addr += l;

    }



    return error;

}
