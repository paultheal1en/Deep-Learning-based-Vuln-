static void io_region_add(MemoryListener *listener,

                          MemoryRegionSection *section)

{

    MemoryRegionIORange *mrio = g_new(MemoryRegionIORange, 1);



    mrio->mr = section->mr;

    mrio->offset = section->offset_within_region;

    iorange_init(&mrio->iorange, &memory_region_iorange_ops,

                 section->offset_within_address_space,

                 int128_get64(section->size));

    ioport_register(&mrio->iorange);

}
