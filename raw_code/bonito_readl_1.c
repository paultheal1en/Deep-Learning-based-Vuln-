static uint64_t bonito_readl(void *opaque, hwaddr addr,

                             unsigned size)

{

    PCIBonitoState *s = opaque;

    uint32_t saddr;



    saddr = (addr - BONITO_REGBASE) >> 2;



    DPRINTF("bonito_readl "TARGET_FMT_plx"\n", addr);

    switch (saddr) {

    case BONITO_INTISR:

        return s->regs[saddr];

    default:

        return s->regs[saddr];

    }

}
