static void kvm_log_stop(MemoryListener *listener,

                          MemoryRegionSection *section)

{

    int r;



    r = kvm_dirty_pages_log_change(section->offset_within_address_space,

                                   int128_get64(section->size), false);

    if (r < 0) {

        abort();

    }

}
