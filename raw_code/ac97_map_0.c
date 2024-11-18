static void ac97_map (PCIDevice *pci_dev, int region_num,

                      uint32_t addr, uint32_t size, int type)

{

    PCIAC97LinkState *d = (PCIAC97LinkState *) pci_dev;

    AC97LinkState *s = &d->ac97;



    if (!region_num) {

        s->base[0] = addr;

        register_ioport_read (addr, 256 * 1, 1, nam_readb, d);

        register_ioport_read (addr, 256 * 2, 2, nam_readw, d);

        register_ioport_read (addr, 256 * 4, 4, nam_readl, d);

        register_ioport_write (addr, 256 * 1, 1, nam_writeb, d);

        register_ioport_write (addr, 256 * 2, 2, nam_writew, d);

        register_ioport_write (addr, 256 * 4, 4, nam_writel, d);

    }

    else {

        s->base[1] = addr;

        register_ioport_read (addr, 64 * 1, 1, nabm_readb, d);

        register_ioport_read (addr, 64 * 2, 2, nabm_readw, d);

        register_ioport_read (addr, 64 * 4, 4, nabm_readl, d);

        register_ioport_write (addr, 64 * 1, 1, nabm_writeb, d);

        register_ioport_write (addr, 64 * 2, 2, nabm_writew, d);

        register_ioport_write (addr, 64 * 4, 4, nabm_writel, d);

    }

}
