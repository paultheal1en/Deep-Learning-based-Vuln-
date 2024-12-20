static uint32_t bonito_readl(void *opaque, target_phys_addr_t addr)

{

    PCIBonitoState *s = opaque;

    uint32_t saddr;



    saddr = (addr - BONITO_REGBASE) >> 2;



    DPRINTF("bonito_readl "TARGET_FMT_plx"  \n", addr);

    switch (saddr) {

    case BONITO_INTISR:

        return s->regs[saddr];

    default:

        return s->regs[saddr];

    }

}
