bool memory_region_get_dirty(MemoryRegion *mr, hwaddr addr,

                             hwaddr size, unsigned client)

{

    assert(mr->terminates);

    return cpu_physical_memory_get_dirty(mr->ram_addr + addr, size, client);

}
