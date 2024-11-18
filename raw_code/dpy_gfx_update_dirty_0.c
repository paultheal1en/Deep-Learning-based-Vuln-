void dpy_gfx_update_dirty(QemuConsole *con,

                          MemoryRegion *address_space,

                          hwaddr base,

                          bool invalidate)

{

    DisplaySurface *ds = qemu_console_surface(con);

    int width = surface_stride(ds);

    int height = surface_height(ds);

    hwaddr size = width * height;

    MemoryRegionSection mem_section;

    MemoryRegion *mem;

    ram_addr_t addr;

    int first, last, i;

    bool dirty;



    mem_section = memory_region_find(address_space, base, size);

    mem = mem_section.mr;

    if (int128_get64(mem_section.size) != size ||

        !memory_region_is_ram(mem_section.mr)) {

        goto out;

    }

    assert(mem);



    memory_region_sync_dirty_bitmap(mem);

    addr = mem_section.offset_within_region;



    first = -1;

    last = -1;

    for (i = 0; i < height; i++, addr += width) {

        dirty = invalidate ||

            memory_region_get_dirty(mem, addr, width, DIRTY_MEMORY_VGA);

        if (dirty) {

            if (first == -1) {

                first = i;

            }

            last = i;

        }

        if (first != -1 && !dirty) {

            assert(last != -1 && last >= first);

            dpy_gfx_update(con, 0, first, surface_width(ds),

                           last - first + 1);

            first = -1;

        }

    }

    if (first != -1) {

        assert(last != -1 && last >= first);

        dpy_gfx_update(con, 0, first, surface_width(ds),

                       last - first + 1);

    }



    memory_region_reset_dirty(mem, mem_section.offset_within_region, size,

                              DIRTY_MEMORY_VGA);

out:

    memory_region_unref(mem);

}
