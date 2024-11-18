static void xen_log_start(MemoryListener *listener,

                          MemoryRegionSection *section)

{

    XenIOState *state = container_of(listener, XenIOState, memory_listener);



    xen_sync_dirty_bitmap(state, section->offset_within_address_space,

                          int128_get64(section->size));

}
