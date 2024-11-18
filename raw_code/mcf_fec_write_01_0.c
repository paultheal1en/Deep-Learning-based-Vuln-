static void mcf_fec_write(void *opaque, hwaddr addr,

                          uint64_t value, unsigned size)

{

    mcf_fec_state *s = (mcf_fec_state *)opaque;

    switch (addr & 0x3ff) {

    case 0x004:

        s->eir &= ~value;

        break;

    case 0x008:

        s->eimr = value;

        break;

    case 0x010: /* RDAR */

        if ((s->ecr & FEC_EN) && !s->rx_enabled) {

            DPRINTF("RX enable\n");

            mcf_fec_enable_rx(s);

        }

        break;

    case 0x014: /* TDAR */

        if (s->ecr & FEC_EN) {

            mcf_fec_do_tx(s);

        }

        break;

    case 0x024:

        s->ecr = value;

        if (value & FEC_RESET) {

            DPRINTF("Reset\n");

            mcf_fec_reset(s);

        }

        if ((s->ecr & FEC_EN) == 0) {

            s->rx_enabled = 0;

        }

        break;

    case 0x040:

        s->mmfr = value;

        s->eir |= FEC_INT_MII;

        break;

    case 0x044:

        s->mscr = value & 0xfe;

        break;

    case 0x064:

        /* TODO: Implement MIB.  */

        break;

    case 0x084:

        s->rcr = value & 0x07ff003f;

        /* TODO: Implement LOOP mode.  */

        break;

    case 0x0c4: /* TCR */

        /* We transmit immediately, so raise GRA immediately.  */

        s->tcr = value;

        if (value & 1)

            s->eir |= FEC_INT_GRA;

        break;

    case 0x0e4: /* PALR */

        s->conf.macaddr.a[0] = value >> 24;

        s->conf.macaddr.a[1] = value >> 16;

        s->conf.macaddr.a[2] = value >> 8;

        s->conf.macaddr.a[3] = value;

        break;

    case 0x0e8: /* PAUR */

        s->conf.macaddr.a[4] = value >> 24;

        s->conf.macaddr.a[5] = value >> 16;

        break;

    case 0x0ec:

        /* OPD */

        break;

    case 0x118:

    case 0x11c:

    case 0x120:

    case 0x124:

        /* TODO: implement MAC hash filtering.  */

        break;

    case 0x144:

        s->tfwr = value & 3;

        break;

    case 0x14c:

        /* FRBR writes ignored.  */

        break;

    case 0x150:

        s->rfsr = (value & 0x3fc) | 0x400;

        break;

    case 0x180:

        s->erdsr = value & ~3;

        s->rx_descriptor = s->erdsr;

        break;

    case 0x184:

        s->etdsr = value & ~3;

        s->tx_descriptor = s->etdsr;

        break;

    case 0x188:

        s->emrbr = value & 0x7f0;

        break;

    default:

        hw_error("mcf_fec_write Bad address 0x%x\n", (int)addr);

    }

    mcf_fec_update(s);

}
