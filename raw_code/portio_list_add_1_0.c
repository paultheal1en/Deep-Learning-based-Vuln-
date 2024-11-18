static void portio_list_add_1(PortioList *piolist,

                              const MemoryRegionPortio *pio_init,

                              unsigned count, unsigned start,

                              unsigned off_low, unsigned off_high)

{

    MemoryRegionPortio *pio;

    MemoryRegionOps *ops;

    MemoryRegion *region, *alias;

    unsigned i;



    /* Copy the sub-list and null-terminate it.  */

    pio = g_new(MemoryRegionPortio, count + 1);

    memcpy(pio, pio_init, sizeof(MemoryRegionPortio) * count);

    memset(pio + count, 0, sizeof(MemoryRegionPortio));



    /* Adjust the offsets to all be zero-based for the region.  */

    for (i = 0; i < count; ++i) {

        pio[i].offset -= off_low;

    }



    ops = g_new0(MemoryRegionOps, 1);

    ops->old_portio = pio;



    region = g_new(MemoryRegion, 1);

    alias = g_new(MemoryRegion, 1);

    /*

     * Use an alias so that the callback is called with an absolute address,

     * rather than an offset relative to to start + off_low.

     */

    memory_region_init_io(region, ops, piolist->opaque, piolist->name,

                          INT64_MAX);

    memory_region_init_alias(alias, piolist->name,

                             region, start + off_low, off_high - off_low);

    memory_region_add_subregion(piolist->address_space,

                                start + off_low, alias);

    piolist->regions[piolist->nr] = region;

    piolist->aliases[piolist->nr] = alias;

    ++piolist->nr;

}
