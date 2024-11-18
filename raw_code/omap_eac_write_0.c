static void omap_eac_write(void *opaque, target_phys_addr_t addr,

                           uint64_t value, unsigned size)

{

    struct omap_eac_s *s = (struct omap_eac_s *) opaque;



    if (size != 2) {

        return omap_badwidth_write16(opaque, addr, value);

    }



    switch (addr) {

    case 0x098:	/* APD1LCR */

    case 0x09c:	/* APD1RCR */

    case 0x0a0:	/* APD2LCR */

    case 0x0a4:	/* APD2RCR */

    case 0x0a8:	/* APD3LCR */

    case 0x0ac:	/* APD3RCR */

    case 0x0b0:	/* APD4R */

    case 0x0b8:	/* ADRDR */

    case 0x0d0:	/* MPDDMARR */

    case 0x0d8:	/* MPUDMARR */

    case 0x0e4:	/* BPDDMARR */

    case 0x0ec:	/* BPUDMARR */

    case 0x100:	/* VERSION_NUMBER */

    case 0x108:	/* SYSSTATUS */

        OMAP_RO_REG(addr);

        return;



    case 0x000:	/* CPCFR1 */

        s->config[0] = value & 0xff;

        omap_eac_format_update(s);

        break;

    case 0x004:	/* CPCFR2 */

        s->config[1] = value & 0xff;

        omap_eac_format_update(s);

        break;

    case 0x008:	/* CPCFR3 */

        s->config[2] = value & 0xff;

        omap_eac_format_update(s);

        break;

    case 0x00c:	/* CPCFR4 */

        s->config[3] = value & 0xff;

        omap_eac_format_update(s);

        break;



    case 0x010:	/* CPTCTL */

        /* Assuming TXF and TXE bits are read-only... */

        s->control = value & 0x5f;

        omap_eac_interrupt_update(s);

        break;



    case 0x014:	/* CPTTADR */

        s->address = value & 0xff;

        break;

    case 0x018:	/* CPTDATL */

        s->data &= 0xff00;

        s->data |= value & 0xff;

        break;

    case 0x01c:	/* CPTDATH */

        s->data &= 0x00ff;

        s->data |= value << 8;

        break;

    case 0x020:	/* CPTVSLL */

        s->vtol = value & 0xf8;

        break;

    case 0x024:	/* CPTVSLH */

        s->vtsl = value & 0x9f;

        break;

    case 0x040:	/* MPCTR */

        s->modem.control = value & 0x8f;

        break;

    case 0x044:	/* MPMCCFR */

        s->modem.config = value & 0x7fff;

        break;

    case 0x060:	/* BPCTR */

        s->bt.control = value & 0x8f;

        break;

    case 0x064:	/* BPMCCFR */

        s->bt.config = value & 0x7fff;

        break;

    case 0x080:	/* AMSCFR */

        s->mixer = value & 0x0fff;

        break;

    case 0x084:	/* AMVCTR */

        s->gain[0] = value & 0xffff;

        break;

    case 0x088:	/* AM1VCTR */

        s->gain[1] = value & 0xff7f;

        break;

    case 0x08c:	/* AM2VCTR */

        s->gain[2] = value & 0xff7f;

        break;

    case 0x090:	/* AM3VCTR */

        s->gain[3] = value & 0xff7f;

        break;

    case 0x094:	/* ASTCTR */

        s->att = value & 0xff;

        break;



    case 0x0b4:	/* ADWR */

        s->codec.txbuf[s->codec.txlen ++] = value;

        if (unlikely(s->codec.txlen == EAC_BUF_LEN ||

                                s->codec.txlen == s->codec.txavail)) {

            if (s->codec.txavail)

                omap_eac_out_empty(s);

            /* Discard what couldn't be written */

            s->codec.txlen = 0;

        }

        break;



    case 0x0bc:	/* AGCFR */

        s->codec.config[0] = value & 0x07ff;

        omap_eac_format_update(s);

        break;

    case 0x0c0:	/* AGCTR */

        s->codec.config[1] = value & 0x780f;

        omap_eac_format_update(s);

        break;

    case 0x0c4:	/* AGCFR2 */

        s->codec.config[2] = value & 0x003f;

        omap_eac_format_update(s);

        break;

    case 0x0c8:	/* AGCFR3 */

        s->codec.config[3] = value & 0xffff;

        omap_eac_format_update(s);

        break;

    case 0x0cc:	/* MBPDMACTR */

    case 0x0d4:	/* MPDDMAWR */

    case 0x0e0:	/* MPUDMAWR */

    case 0x0e8:	/* BPDDMAWR */

    case 0x0f0:	/* BPUDMAWR */

        break;



    case 0x104:	/* SYSCONFIG */

        if (value & (1 << 1))				/* SOFTRESET */

            omap_eac_reset(s);

        s->sysconfig = value & 0x31d;

        break;



    default:

        OMAP_BAD_REG(addr);

        return;

    }

}
