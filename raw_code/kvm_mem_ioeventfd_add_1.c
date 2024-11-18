static void kvm_mem_ioeventfd_add(MemoryListener *listener,

                                  MemoryRegionSection *section,

                                  bool match_data, uint64_t data,

                                  EventNotifier *e)

{

    int fd = event_notifier_get_fd(e);

    int r;



    r = kvm_set_ioeventfd_mmio(fd, section->offset_within_address_space,

                               data, true, int128_get64(section->size),

                               match_data);

    if (r < 0) {



        abort();

    }

}