static void memory_region_update_coalesced_range_as(MemoryRegion *mr, AddressSpace *as)

{

    FlatView *view;

    FlatRange *fr;

    CoalescedMemoryRange *cmr;

    AddrRange tmp;

    MemoryRegionSection section;



    view = as->current_map;

    FOR_EACH_FLAT_RANGE(fr, view) {

        if (fr->mr == mr) {

            section = (MemoryRegionSection) {

                .address_space = as,

                .offset_within_address_space = int128_get64(fr->addr.start),

                .size = fr->addr.size,

            };



            MEMORY_LISTENER_CALL(coalesced_mmio_del, Reverse, &section,

                                 int128_get64(fr->addr.start),

                                 int128_get64(fr->addr.size));

            QTAILQ_FOREACH(cmr, &mr->coalesced, link) {

                tmp = addrrange_shift(cmr->addr,

                                      int128_sub(fr->addr.start,

                                                 int128_make64(fr->offset_in_region)));

                if (!addrrange_intersects(tmp, fr->addr)) {

                    continue;

                }

                tmp = addrrange_intersection(tmp, fr->addr);

                MEMORY_LISTENER_CALL(coalesced_mmio_add, Forward, &section,

                                     int128_get64(tmp.start),

                                     int128_get64(tmp.size));

            }

        }

    }

}
