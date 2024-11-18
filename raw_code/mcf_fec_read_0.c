static uint64_t mcf_fec_read(void *opaque, target_phys_addr_t addr,

                             unsigned size)

{

    mcf_fec_state *s = (mcf_fec_state *)opaque;

    switch (addr & 0x3ff) {

    case 0x004: return s->eir;

    case 0x008: return s->eimr;

    case 0x010: return s->rx_enabled ? (1 << 24) : 0; /* RDAR */

    case 0x014: return 0; /* TDAR */

    case 0x024: return s->ecr;

    case 0x040: return s->mmfr;

    case 0x044: return s->mscr;

    case 0x064: return 0; /* MIBC */

    case 0x084: return s->rcr;

    case 0x0c4: return s->tcr;

    case 0x0e4: /* PALR */

        return (s->conf.macaddr.a[0] << 24) | (s->conf.macaddr.a[1] << 16)

              | (s->conf.macaddr.a[2] << 8) | s->conf.macaddr.a[3];

        break;

    case 0x0e8: /* PAUR */

        return (s->conf.macaddr.a[4] << 24) | (s->conf.macaddr.a[5] << 16) | 0x8808;

    case 0x0ec: return 0x10000; /* OPD */

    case 0x118: return 0;

    case 0x11c: return 0;

    case 0x120: return 0;

    case 0x124: return 0;

    case 0x144: return s->tfwr;

    case 0x14c: return 0x600;

    case 0x150: return s->rfsr;

    case 0x180: return s->erdsr;

    case 0x184: return s->etdsr;

    case 0x188: return s->emrbr;

    default:

        hw_error("mcf_fec_read: Bad address 0x%x\n", (int)addr);

        return 0;

    }

}
