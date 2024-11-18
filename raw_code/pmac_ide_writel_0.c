static void pmac_ide_writel (void *opaque,

                             target_phys_addr_t addr, uint32_t val)

{

    MACIOIDEState *d = opaque;



    addr = (addr & 0xFFF) >> 4;

    val = bswap32(val);

    if (addr == 0) {

        ide_data_writel(&d->bus, 0, val);

    }

}
