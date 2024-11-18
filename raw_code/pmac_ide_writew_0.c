static void pmac_ide_writew (void *opaque,

                             target_phys_addr_t addr, uint32_t val)

{

    MACIOIDEState *d = opaque;



    addr = (addr & 0xFFF) >> 4;

    val = bswap16(val);

    if (addr == 0) {

        ide_data_writew(&d->bus, 0, val);

    }

}
