static uint32_t pmac_ide_readl (void *opaque,target_phys_addr_t addr)

{

    uint32_t retval;

    MACIOIDEState *d = opaque;



    addr = (addr & 0xFFF) >> 4;

    if (addr == 0) {

        retval = ide_data_readl(&d->bus, 0);

    } else {

        retval = 0xFFFFFFFF;

    }

    retval = bswap32(retval);

    return retval;

}
