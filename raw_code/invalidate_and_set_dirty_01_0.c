static void invalidate_and_set_dirty(MemoryRegion *mr, hwaddr addr,

                                     hwaddr length)

{

    if (cpu_physical_memory_range_includes_clean(addr, length)) {

        uint8_t dirty_log_mask = memory_region_get_dirty_log_mask(mr);

        if (dirty_log_mask & (1 << DIRTY_MEMORY_CODE)) {

            tb_invalidate_phys_range(addr, addr + length);

            dirty_log_mask &= ~(1 << DIRTY_MEMORY_CODE);

        }

        cpu_physical_memory_set_dirty_range(addr, length, dirty_log_mask);

    } else {

        xen_modified_memory(addr, length);

    }

}
