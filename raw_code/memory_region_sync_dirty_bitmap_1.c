void memory_region_sync_dirty_bitmap(MemoryRegion *mr)

{

    FlatRange *fr;



    FOR_EACH_FLAT_RANGE(fr, &address_space_memory.current_map) {

        if (fr->mr == mr) {

            MEMORY_LISTENER_UPDATE_REGION(fr, &address_space_memory,

                                          Forward, log_sync);

        }

    }

}
