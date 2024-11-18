void portio_list_add(PortioList *piolist,

                     MemoryRegion *address_space,

                     uint32_t start)

{

    const MemoryRegionPortio *pio, *pio_start = piolist->ports;

    unsigned int off_low, off_high, off_last, count;



    piolist->address_space = address_space;



    /* Handle the first entry specially.  */

    off_last = off_low = pio_start->offset;

    off_high = off_low + pio_start->len;

    count = 1;



    for (pio = pio_start + 1; pio->size != 0; pio++, count++) {

        /* All entries must be sorted by offset.  */

        assert(pio->offset >= off_last);

        off_last = pio->offset;



        /* If we see a hole, break the region.  */

        if (off_last > off_high) {

            portio_list_add_1(piolist, pio_start, count, start, off_low,

                              off_high);

            /* ... and start collecting anew.  */

            pio_start = pio;

            off_low = off_last;

            off_high = off_low + pio->len;

            count = 0;

        } else if (off_last + pio->len > off_high) {

            off_high = off_last + pio->len;

        }

    }



    /* There will always be an open sub-list.  */

    portio_list_add_1(piolist, pio_start, count, start, off_low, off_high);

}
