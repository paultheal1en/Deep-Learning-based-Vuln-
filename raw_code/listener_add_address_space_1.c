static void listener_add_address_space(MemoryListener *listener,

                                       AddressSpace *as)

{

    FlatRange *fr;



    if (listener->address_space_filter

        && listener->address_space_filter != as->root) {

        return;

    }



    if (global_dirty_log) {

        listener->log_global_start(listener);

    }

    FOR_EACH_FLAT_RANGE(fr, &as->current_map) {

        MemoryRegionSection section = {

            .mr = fr->mr,

            .address_space = as->root,

            .offset_within_region = fr->offset_in_region,

            .size = int128_get64(fr->addr.size),

            .offset_within_address_space = int128_get64(fr->addr.start),

            .readonly = fr->readonly,

        };

        listener->region_add(listener, &section);

    }

}
