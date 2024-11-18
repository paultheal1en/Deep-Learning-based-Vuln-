void address_space_sync_dirty_bitmap(AddressSpace *as)

{

    FlatView *view;

    FlatRange *fr;



    view = as->current_map;

    FOR_EACH_FLAT_RANGE(fr, view) {

        MEMORY_LISTENER_UPDATE_REGION(fr, as, Forward, log_sync);

    }

}
