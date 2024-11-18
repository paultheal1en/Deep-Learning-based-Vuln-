static void pcnet_ioport_write(void *opaque, target_phys_addr_t addr,

                               uint64_t data, unsigned size)

{

    PCNetState *d = opaque;



    if (addr < 16 && size == 1) {

        return pcnet_aprom_writeb(d, addr, data);

    } else if (addr >= 0x10 && addr < 0x20 && size == 2) {

        return pcnet_ioport_writew(d, addr, data);

    } else if (addr >= 0x10 && addr < 0x20 && size == 4) {

        return pcnet_ioport_writel(d, addr, data);

    }

}
