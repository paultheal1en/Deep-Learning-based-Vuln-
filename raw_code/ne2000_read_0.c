static uint64_t ne2000_read(void *opaque, target_phys_addr_t addr,

                            unsigned size)

{

    NE2000State *s = opaque;



    if (addr < 0x10 && size == 1) {

        return ne2000_ioport_read(s, addr);

    } else if (addr == 0x10) {

        if (size <= 2) {

            return ne2000_asic_ioport_read(s, addr);

        } else {

            return ne2000_asic_ioport_readl(s, addr);

        }

    } else if (addr == 0x1f && size == 1) {

        return ne2000_reset_ioport_read(s, addr);

    }

    return ((uint64_t)1 << (size * 8)) - 1;

}
