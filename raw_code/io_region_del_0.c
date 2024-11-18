static void io_region_del(MemoryListener *listener,

                          MemoryRegionSection *section)

{

    isa_unassign_ioport(section->offset_within_address_space,

                        int128_get64(section->size));

}
