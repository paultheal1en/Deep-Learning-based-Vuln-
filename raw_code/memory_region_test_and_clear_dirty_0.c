bool memory_region_test_and_clear_dirty(MemoryRegion *mr, hwaddr addr,

                                        hwaddr size, unsigned client)

{

    assert(mr->terminates);

    return cpu_physical_memory_test_and_clear_dirty(mr->ram_addr + addr,

                                                    size, client);

}
