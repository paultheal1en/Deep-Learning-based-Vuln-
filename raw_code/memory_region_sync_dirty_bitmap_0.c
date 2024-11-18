void memory_region_sync_dirty_bitmap(MemoryRegion *mr)

{

    AddressSpace *as;

    FlatRange *fr;



    QTAILQ_FOREACH(as, &address_spaces, address_spaces_link) {

        FlatView *view = as->current_map;

        FOR_EACH_FLAT_RANGE(fr, view) {

            if (fr->mr == mr) {

                MEMORY_LISTENER_UPDATE_REGION(fr, as, Forward, log_sync);

            }

        }

    }

}
