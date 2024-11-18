static void bonito_writel(void *opaque, hwaddr addr,

                          uint64_t val, unsigned size)

{

    PCIBonitoState *s = opaque;

    uint32_t saddr;

    int reset = 0;



    saddr = (addr - BONITO_REGBASE) >> 2;



    DPRINTF("bonito_writel "TARGET_FMT_plx" val %x saddr %x\n", addr, val, saddr);

    switch (saddr) {

    case BONITO_BONPONCFG:

    case BONITO_IODEVCFG:

    case BONITO_SDCFG:

    case BONITO_PCIMAP:

    case BONITO_PCIMEMBASECFG:

    case BONITO_PCIMAP_CFG:

    case BONITO_GPIODATA:

    case BONITO_GPIOIE:

    case BONITO_INTEDGE:

    case BONITO_INTSTEER:

    case BONITO_INTPOL:

    case BONITO_PCIMAIL0:

    case BONITO_PCIMAIL1:

    case BONITO_PCIMAIL2:

    case BONITO_PCIMAIL3:

    case BONITO_PCICACHECTRL:

    case BONITO_PCICACHETAG:

    case BONITO_PCIBADADDR:

    case BONITO_PCIMSTAT:

    case BONITO_TIMECFG:

    case BONITO_CPUCFG:

    case BONITO_DQCFG:

    case BONITO_MEMSIZE:

        s->regs[saddr] = val;

        break;

    case BONITO_BONGENCFG:

        if (!(s->regs[saddr] & 0x04) && (val & 0x04)) {

            reset = 1; /* bit 2 jump from 0 to 1 cause reset */

        }

        s->regs[saddr] = val;

        if (reset) {

            qemu_system_reset_request();

        }

        break;

    case BONITO_INTENSET:

        s->regs[BONITO_INTENSET] = val;

        s->regs[BONITO_INTEN] |= val;

        break;

    case BONITO_INTENCLR:

        s->regs[BONITO_INTENCLR] = val;

        s->regs[BONITO_INTEN] &= ~val;

        break;

    case BONITO_INTEN:

    case BONITO_INTISR:

        DPRINTF("write to readonly bonito register %x\n", saddr);

        break;

    default:

        DPRINTF("write to unknown bonito register %x\n", saddr);

        break;

    }

}
