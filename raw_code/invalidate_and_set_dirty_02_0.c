static void invalidate_and_set_dirty(hwaddr addr,

                                     hwaddr length)

{

    if (cpu_physical_memory_range_includes_clean(addr, length)) {

        tb_invalidate_phys_range(addr, addr + length, 0);

        cpu_physical_memory_set_dirty_range_nocode(addr, length);

    }

    xen_modified_memory(addr, length);

}
