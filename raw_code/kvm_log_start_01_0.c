static void kvm_log_start(MemoryListener *listener,

                          MemoryRegionSection *section)

{

    int r;



    r = kvm_dirty_pages_log_change(section->offset_within_address_space,

                                   int128_get64(section->size), true);

    if (r < 0) {

        abort();

    }

}
