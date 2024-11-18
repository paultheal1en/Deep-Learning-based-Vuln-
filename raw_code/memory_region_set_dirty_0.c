void memory_region_set_dirty(MemoryRegion *mr, hwaddr addr,

                             hwaddr size)

{

    assert(mr->terminates);

    cpu_physical_memory_set_dirty_range(mr->ram_addr + addr, size,

                                        memory_region_get_dirty_log_mask(mr));

}
