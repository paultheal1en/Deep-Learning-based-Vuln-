void cpu_physical_memory_unmap(void *buffer, target_phys_addr_t len,

                               int is_write, target_phys_addr_t access_len)

{

    if (buffer != bounce.buffer) {

        if (is_write) {

            ram_addr_t addr1 = qemu_ram_addr_from_host(buffer);

            while (access_len) {

                unsigned l;

                l = TARGET_PAGE_SIZE;

                if (l > access_len)

                    l = access_len;

                if (!cpu_physical_memory_is_dirty(addr1)) {

                    /* invalidate code */

                    tb_invalidate_phys_page_range(addr1, addr1 + l, 0);

                    /* set dirty bit */

                    phys_ram_dirty[addr1 >> TARGET_PAGE_BITS] |=

                        (0xff & ~CODE_DIRTY_FLAG);

                }

                addr1 += l;

                access_len -= l;

            }

        }

        return;

    }

    if (is_write) {

        cpu_physical_memory_write(bounce.addr, bounce.buffer, access_len);

    }

    qemu_vfree(bounce.buffer);

    bounce.buffer = NULL;

    cpu_notify_map_clients();

}
