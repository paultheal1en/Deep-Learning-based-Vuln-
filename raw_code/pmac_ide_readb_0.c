static uint32_t pmac_ide_readb (void *opaque,target_phys_addr_t addr)

{

    uint8_t retval;

    MACIOIDEState *d = opaque;



    addr = (addr & 0xFFF) >> 4;

    switch (addr) {

    case 1 ... 7:

        retval = ide_ioport_read(&d->bus, addr);

        break;

    case 8:

    case 22:

        retval = ide_status_read(&d->bus, 0);

        break;

    default:

        retval = 0xFF;

        break;

    }

    return retval;

}
